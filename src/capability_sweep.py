"""Capability vs steering strength: teacher-forced NLL and next-token top-1 accuracy.

Concept and fluency metrics (metrics.py) are both about the generated continuation, which is itself
produced under the intervention -- a model that has started to degenerate writes its own evidence.
This script instead holds a fixed batch of ordinary text fixed (held-out OpenWebText documents, the
same corpus and shard the generation prompts come from, but never seen by them: prompts come from
document indices < a few hundred, these start at index 3000) and asks how well the model predicts
*that* text while the same steering hook used for generation is active. That separates "the model's
underlying next-token competence degraded" from "the model said something off-topic."

Arms and per-feature arm construction are imported from generate.py (build_arms / resolve_per_feature)
rather than reimplemented, so this always sees the same intervention code the generation pipeline
uses. If an arm name isn't recognised yet (e.g. one still being added by another executor), that arm
is skipped with a printed message rather than aborting the whole sweep.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from common import (
    DEVICE,
    HOOK,
    RESULTS,
    ROOT,
    ActStats,
    load_ceilings,
    load_features,
    load_model,
    load_sae,
    natural_strength,
    seed_all,
)
from generate import build_arms, resolve_per_feature
from steering import HookState, make_hook

# Shard 1 of OpenWebText: the same shard src/activations.py draws generation prompts from (its first
# ~2000 documents). Starting at document 3000 keeps this well clear of that range.
PROMPT_SHARD = "hf://datasets/Skylion007/openwebtext/plain_text/train-00001-of-00080.parquet"
START_DOC = 3000


def load_docs(model, n_docs: int, ctx: int, start_doc: int = START_DOC) -> torch.Tensor:
    """`n_docs` documents that tokenise to at least `ctx` tokens, truncated to exactly `ctx`.

    Mirrors activations.py's `_iter_batches`: truncate raw text to 4000 chars before tokenising (BPE
    is slow on long strings and 4000 chars is already far more than `ctx` tokens need), then keep only
    documents whose token count reaches `ctx` -- a short document would otherwise force padding, and
    padding would have to be masked out of the NLL/accuracy average.
    """
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=PROMPT_SHARD, split="train")
    toks = []
    i = start_doc
    n_avail = len(ds)
    while len(toks) < n_docs and i < n_avail:
        text = ds[i]["text"]
        i += 1
        if len(text) < 400:
            continue
        t = model.to_tokens(text[:4000], truncate=True)[0, :ctx]
        if t.shape[0] < ctx:
            continue
        toks.append(t)
    if len(toks) < n_docs:
        raise RuntimeError(
            f"only found {len(toks)}/{n_docs} documents of >= {ctx} tokens starting at index {start_doc} "
            f"(scanned up to index {i} of {n_avail})"
        )
    print(f"loaded {len(toks)} documents x {ctx} tokens from shard doc index {start_doc}..{i - 1}")
    return torch.stack(toks)


@torch.no_grad()
def eval_arm(model, docs: torch.Tensor, fn, v_hat: torch.Tensor, s: float, batch: int) -> tuple[float, float]:
    """Teacher-forced mean NLL (nats) and next-token top-1 accuracy over all predicted positions.

    One forward pass per batch (no KV cache, no generation), so HookState's position offset starts at
    0 for every batch and make_hook's SKIP_POS masking applies exactly as it does on the first pass of
    generation.
    """
    total_nll, total_correct, total_n = 0.0, 0.0, 0
    for i in range(0, docs.shape[0], batch):
        chunk = docs[i : i + batch].to(DEVICE)
        state = HookState()
        with model.hooks(fwd_hooks=[(HOOK, make_hook(fn, v_hat, s, state))]):
            logits = model(chunk)
        logits = logits[:, :-1].float()
        tgt = chunk[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        correct = (logits.argmax(-1) == tgt).float()
        total_nll += float(nll.sum())
        total_correct += float(correct.sum())
        total_n += int(nll.numel())
        del logits, logp, nll, correct
    return total_nll / total_n, total_correct / total_n


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for row in df.itertuples(index=False):
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def main(args) -> None:
    seed_all(args.seed)
    t0 = time.time()
    model = load_model()
    sae = load_sae()
    stats = ActStats.load()
    frozen = yaml.safe_load((ROOT / "configs" / "frozen.yaml").read_text(encoding="utf-8"))
    for kv in args.set or []:
        k, v = kv.split('=', 1)
        frozen[k] = yaml.safe_load(v)
    feats = load_features(args.split)
    feats = feats[: args.features]
    ceilings = load_ceilings()
    grid = [float(c) for c in args.c_grid.split(",")]

    docs = load_docs(model, args.n_docs, args.ctx)

    requested_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    skipped_arms = []
    results: dict[tuple[str, float], list[tuple[float, float]]] = {}

    for arm_name in requested_arms:
        try:
            built = build_arms([arm_name], model, sae, stats, frozen)
        except Exception as e:
            print(f"skipping arm '{arm_name}': missing dependency ({type(e).__name__}: {e})")
            skipped_arms.append(arm_name)
            continue
        arm = built[arm_name]

        for rec in tqdm(feats, desc=arm_name, unit="feature"):
            f = int(rec["index"])
            v = sae.W_dec[f].detach().float()
            v_hat = v / v.norm()
            scale = natural_strength(sae, ceilings, f)
            fn = arm
            if isinstance(arm, tuple):
                try:
                    fn = resolve_per_feature(arm[1], sae, stats, frozen, v_hat, f)
                except Exception as e:
                    print(f"skipping arm '{arm_name}' feature {f}: missing dependency ({type(e).__name__}: {e})")
                    continue
            for c in grid:
                s = c * scale
                nll, acc = eval_arm(model, docs, fn, v_hat, s, args.batch)
                results.setdefault((arm_name, c), []).append((nll, acc))

    rows = []
    for (arm_name, c), vals in results.items():
        if not vals:
            continue
        rows.append(
            {
                "arm": arm_name,
                "c": c,
                "nll": float(np.mean([v[0] for v in vals])),
                "top1_acc": float(np.mean([v[1] for v in vals])),
                "n_features": len(vals),
            }
        )
    df = pd.DataFrame(rows).sort_values(["arm", "c"]).reset_index(drop=True)
    df["nll_delta_vs_c0"] = np.nan
    df["acc_delta_vs_c0"] = np.nan
    for arm_name, g in df.groupby("arm"):
        base = g[np.isclose(g["c"], 0.0)]
        if base.empty:
            continue
        base_nll, base_acc = float(base["nll"].iloc[0]), float(base["top1_acc"].iloc[0])
        df.loc[g.index, "nll_delta_vs_c0"] = g["nll"] - base_nll
        df.loc[g.index, "acc_delta_vs_c0"] = g["top1_acc"] - base_acc

    out_path = RESULTS / args.out
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path} in {(time.time() - t0) / 60:.1f} min")

    lines = [
        "# Capability vs steering strength",
        "",
        f"Held out documents: {args.n_docs} x {args.ctx} tokens from `{Path(PROMPT_SHARD).name}`, "
        f"starting at document index {START_DOC}. Features: first {args.features} of split `{args.split}`.",
        "",
    ]
    if skipped_arms:
        lines.append(f"Arms skipped (missing dependency): {', '.join(skipped_arms)}")
        lines.append("")
    lines.append("## nll and top1_acc by arm x c")
    lines.append("")
    lines.append(_df_to_md(df.round(4)) if not df.empty else "(no data)")
    lines.append("")
    lines.append("## first c where top1_acc drops more than 5 points (0.05) from c=0, per arm")
    lines.append("")
    drop_rows = []
    for arm_name, g in df.sort_values("c").groupby("arm"):
        hit = g[g["acc_delta_vs_c0"] <= -0.05]
        c_hit = float(hit["c"].iloc[0]) if not hit.empty else None
        drop_rows.append(
            {
                "arm": arm_name,
                "first_c_with_drop_gt_5pt": c_hit if c_hit is not None else "none in grid",
            }
        )
    lines.append(_df_to_md(pd.DataFrame(drop_rows)) if drop_rows else "(no data)")
    lines.append("")
    note_path = RESULTS / "capability_NOTE.md"
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {note_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test_r3", choices=["test", "dev", "test_r3", "test_r4"])
    ap.add_argument("--arms", default="naive,dirfix,shared,shared_only,rotate")
    ap.add_argument("--c-grid", dest="c_grid", default="0,0.5,1,1.5,2,3,4,5")
    ap.add_argument("--features", type=int, default=4)
    ap.add_argument("--n-docs", dest="n_docs", type=int, default=200)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], help="override a frozen.yaml key, e.g. direction=dir_ent")
    ap.add_argument("--out", default="capability_sweep.csv")
    main(ap.parse_args())
