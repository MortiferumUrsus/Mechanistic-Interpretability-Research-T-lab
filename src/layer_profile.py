"""Where does the norm blow up? Per-layer resid_post norm, projection onto d_bar, and final entropy.

A steering intervention lands at HOOK (blocks.6.hook_resid_post) but its effect propagates through
the remaining six layers of the network. This script reads out blocks.{6..11}.hook_resid_post at the
last prompt position under the same generation hook (steering.make_hook at HOOK), for a few arms and
two strengths, and reports how the residual norm and the projection onto the shared direction d_bar
evolve layer by layer, alongside the entropy of the model's final next-token distribution -- one
scalar per (arm, c) attached to every layer row of that cell for easy joint reading.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from common import (
    DATA,
    DEVICE,
    HOOK,
    RESULTS,
    ROOT,
    SHARED_DIRECTION,
    ActStats,
    load_ceilings,
    load_features,
    load_model,
    load_sae,
    natural_strength,
    seed_all,
)
from generate import build_arms, load_prompts
from steering import HookState, make_hook

LAYERS = list(range(6, 12))
LAYER_NAMES = [f"blocks.{L}.hook_resid_post" for L in LAYERS]


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for row in df.itertuples(index=False):
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


@torch.no_grad()
def layer_stats(model, toks: torch.Tensor, fwd_hooks: list, d_bar: torch.Tensor) -> tuple[dict, float]:
    """Per-layer (mean last-position norm, mean last-position projection on d_bar), plus mean entropy
    of the final-position output distribution -- all averaged over the batch of prompts."""
    with model.hooks(fwd_hooks=fwd_hooks):
        logits, cache = model.run_with_cache(toks, names_filter=lambda n: n in LAYER_NAMES)
    out = {}
    for L, name in zip(LAYERS, LAYER_NAMES):
        h_last = cache[name][:, -1, :].float()
        out[L] = (float(h_last.norm(dim=-1).mean()), float((h_last @ d_bar).mean()))
    final_logits = logits[:, -1, :].float()
    p = torch.softmax(final_logits, dim=-1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
    return out, float(ent.mean())


def main(args) -> None:
    seed_all(args.seed)
    model = load_model()
    sae = load_sae()
    stats = ActStats.load()
    frozen = yaml.safe_load((ROOT / "configs" / "frozen.yaml").read_text(encoding="utf-8"))
    d_bar = torch.load(SHARED_DIRECTION, map_location=DEVICE)["d_bar"].to(DEVICE).float()
    feats = load_features(args.split)
    feats = feats[: args.features]
    ceilings = load_ceilings()
    grid = [float(c) for c in args.c_grid.split(",")]

    prompts = load_prompts(model, path=Path(args.prompts))[: args.n_prompts]
    toks_all = model.to_tokens(prompts).to(DEVICE)
    print(f"loaded {len(prompts)} prompts x {toks_all.shape[1]} tokens")

    rows = []

    # Clean baseline: no steering at all, so it does not depend on feature or arm.
    clean_layers, clean_ent = layer_stats(model, toks_all, [], d_bar)
    for L in LAYERS:
        norm, proj = clean_layers[L]
        rows.append({"arm": "clean", "c": 0.0, "layer": L, "norm": norm, "proj_dbar": proj, "entropy": clean_ent})

    requested_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    skipped_arms = []
    for arm_name in requested_arms:
        try:
            built = build_arms([arm_name], model, sae, stats, frozen)
        except Exception as e:
            print(f"skipping arm '{arm_name}': missing dependency ({type(e).__name__}: {e})")
            skipped_arms.append(arm_name)
            continue
        arm = built[arm_name]
        if isinstance(arm, tuple):
            print(f"skipping arm '{arm_name}': per-feature arms ({arm[1]}) are not wired into this script")
            skipped_arms.append(arm_name)
            continue

        for c in tqdm(grid, desc=arm_name, unit="c"):
            per_layer = {L: [] for L in LAYERS}
            ents = []
            for rec in feats:
                f = int(rec["index"])
                v = sae.W_dec[f].detach().float()
                v_hat = v / v.norm()
                scale = natural_strength(sae, ceilings, f)
                s = c * scale
                state = HookState()
                layers, ent = layer_stats(model, toks_all, [(HOOK, make_hook(arm, v_hat, s, state))], d_bar)
                for L in LAYERS:
                    per_layer[L].append(layers[L])
                ents.append(ent)
            mean_ent = float(np.mean(ents))
            for L in LAYERS:
                norms = [x[0] for x in per_layer[L]]
                projs = [x[1] for x in per_layer[L]]
                rows.append(
                    {
                        "arm": arm_name,
                        "c": c,
                        "layer": L,
                        "norm": float(np.mean(norms)),
                        "proj_dbar": float(np.mean(projs)),
                        "entropy": mean_ent,
                    }
                )

    df = pd.DataFrame(rows)
    out_path = RESULTS / "layer_profile.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")

    lines = [
        "# Layer profile: residual norm, projection on d_bar, and final entropy by layer",
        "",
        f"Features: first {args.features} of split `{args.split}`. Prompts: {args.n_prompts} from "
        f"`{args.prompts}`. Strengths: {grid}.",
        "",
    ]
    if skipped_arms:
        lines.append(f"Arms skipped (missing dependency or unsupported): {', '.join(skipped_arms)}")
        lines.append("")
    piv_norm = df.pivot_table(index=["arm", "c"], columns="layer", values="norm").reset_index()
    piv_norm.columns = [f"layer_{c}" if isinstance(c, (int, np.integer)) else c for c in piv_norm.columns]
    lines.append("## mean last-position resid_post norm, by layer (6..11)")
    lines.append("")
    lines.append(_df_to_md(piv_norm.round(3)))
    lines.append("")

    piv_proj = df.pivot_table(index=["arm", "c"], columns="layer", values="proj_dbar").reset_index()
    piv_proj.columns = [f"layer_{c}" if isinstance(c, (int, np.integer)) else c for c in piv_proj.columns]
    lines.append("## mean last-position projection onto d_bar, by layer (6..11)")
    lines.append("")
    lines.append(_df_to_md(piv_proj.round(3)))
    lines.append("")

    ent_tab = df.drop_duplicates(["arm", "c"])[["arm", "c", "entropy"]].reset_index(drop=True)
    lines.append("## entropy of the final next-token distribution, by arm x c")
    lines.append("")
    lines.append(_df_to_md(ent_tab.round(4)))
    lines.append("")

    note_path = RESULTS / "layer_profile_NOTE.md"
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {note_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test_r3", choices=["test", "dev", "test_r3", "test_r4"])
    ap.add_argument("--arms", default="naive,shared,shared_only")
    ap.add_argument("--c-grid", dest="c_grid", default="1,2")
    ap.add_argument("--features", type=int, default=4)
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=30)
    ap.add_argument("--prompts", default=str(DATA / "prompts.json"))
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
