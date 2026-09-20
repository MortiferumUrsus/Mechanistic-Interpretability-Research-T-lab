"""Experiment G runner: concept-conditional denoiser arms (see src/concept_data.py,
src/denoiser_cond.py, src/train_denoiser_cond.py).

Stages, in default order:
    mine        -- concept_data.py mine, then concept_data.py stats.
    pairs_check -- self-check of pair quality on a handful of (feature, position) examples,
                   before any training or generation spend. Needs data/concept_positions.npz
                   (from `mine`) to already exist.
    wiener      -- generate + score + pair the untrained ConditionalWienerArm against naive,
                   on test_r3 (round three, never used to build training pairs or select
                   hyper-parameters for anything).
    train       -- train_denoiser_cond.py (the learned arm).
    learned     -- same as `wiener` but for the trained ConditionalDenoiserArm.
    note        -- results/expG_NOTE.md: arm x strength table from both scored files.

Each stage runs in its own try/except: one stage's failure is printed (with a traceback) and
does not stop the rest.

    python scripts/run_expG.py                       # all stages
    python scripts/run_expG.py --stages pairs_check   # just one
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

STAGES = ["mine", "pairs_check", "wiener", "train", "learned", "note"]
C_GRID = "0,0.5,1.0,1.5,2.0,3.0"


def _run(python: str, args: list[str]) -> None:
    print(f"  $ {' '.join(args)}", flush=True)
    r = subprocess.run([python] + args, cwd=SRC)
    if r.returncode:
        raise RuntimeError(f"{args[0]} exited with code {r.returncode}")


def stage_mine(python: str) -> None:
    _run(python, ["concept_data.py", "mine", "--n-features", "4000", "--top-k", "128", "--chunk", "4096", "--seed", "0"])
    _run(python, ["concept_data.py", "stats"])


def stage_pairs_check(python: str) -> None:
    """Print, for 8 random mined features x 4 positions each: the left context, a_f, ||h+||,
    ||h-||, cos(h+ - h-, W_dec[f]) (must be 1.0 by construction) and the fraction of norm
    removed. A sanity check on the counterfactual-pair arithmetic itself, not on training.
    """
    import numpy as np
    import torch

    from common import DATA, RESULTS, load_model, load_sae, open_memmap

    blob = np.load(DATA / "concept_positions.npz")
    features, positions, acts = blob["features"], blob["positions"], blob["acts"]
    n_feat, top_k = positions.shape

    rng = np.random.default_rng(0)
    feat_pick = rng.choice(n_feat, size=min(8, n_feat), replace=False)

    sae = load_sae()
    model = load_model()
    acts_mm = open_memmap()
    toks_mm = np.memmap(DATA / "toks.i32", dtype=np.int32, mode="r")

    lines = []
    for fi in feat_pick:
        f = int(features[fi])
        w = sae.W_dec[f].detach().float().cpu()
        pos_pick = rng.choice(top_k, size=min(4, top_k), replace=False)
        lines.append(f"=== feature {f} (||W_dec||={float(w.norm()):.4f}) ===")
        for pi in pos_pick:
            p = int(positions[fi, pi])
            a_f = float(acts[fi, pi])
            if p < 0:
                lines.append(f"  slot {pi}: INVALID (sentinel; feature never reached top_k positive activations)")
                continue
            lo = max(0, p - 6)
            ctx_ids = torch.as_tensor(toks_mm[lo : p + 1].astype(np.int64))
            context = model.to_string(ctx_ids)
            target_tok = model.to_string(torch.as_tensor([int(toks_mm[p])]))
            h_plus = torch.from_numpy(np.ascontiguousarray(acts_mm[p])).float()
            h_minus = h_plus - a_f * w
            delta = h_plus - h_minus
            cos = float((delta @ w) / (delta.norm() * w.norm()).clamp_min(1e-8))
            frac = float(delta.norm() / h_plus.norm().clamp_min(1e-8))
            lines.append(
                f"  pos {p}: context={context!r} target_token={target_tok!r} a_f={a_f:.4f} "
                f"||h+||={float(h_plus.norm()):.3f} ||h-||={float(h_minus.norm()):.3f} "
                f"cos(delta,W_dec[f])={cos:.6f} frac_norm_removed={frac:.4f}"
            )
    out_path = RESULTS / "expG_pairs_check.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


def stage_wiener(python: str) -> None:
    _run(python, [
        "generate.py", "--split", "test_r3", "--arms", "naive,cond_wiener",
        "--c-grid", C_GRID, "--n-prompts", "30", "--out", "gen_expG_wiener.jsonl",
    ])
    _run(python, [
        "metrics.py", "--gen", "gen_expG_wiener.jsonl", "--out", "scored_expG_wiener.csv",
        "--split", "test_r3", "--stages", "ppl,keyword,sae,dist", "--shuffle-frac", "0.15",
    ])
    _run(python, [
        "paired_at_strength.py", "--scored", "scored_expG_wiener.csv", "--out", "paired_expG_wiener.csv",
    ])


def stage_train(python: str) -> None:
    _run(python, ["train_denoiser_cond.py", "--steps", "60000", "--name", "cond_mlp_s0"])


def stage_learned(python: str) -> None:
    _run(python, [
        "generate.py", "--split", "test_r3", "--arms", "naive,cond_denoise",
        "--c-grid", C_GRID, "--n-prompts", "30", "--out", "gen_expG_learned.jsonl",
    ])
    _run(python, [
        "metrics.py", "--gen", "gen_expG_learned.jsonl", "--out", "scored_expG_learned.csv",
        "--split", "test_r3", "--stages", "ppl,keyword,sae,dist", "--shuffle-frac", "0.15",
    ])
    _run(python, [
        "paired_at_strength.py", "--scored", "scored_expG_learned.csv", "--out", "paired_expG_learned.csv",
    ])


def stage_note(python: str) -> None:
    """results/expG_NOTE.md: arm x strength table, facts only, from both scored files."""
    import pandas as pd

    from common import RESULTS

    frames = []
    for name in ("scored_expG_wiener.csv", "scored_expG_learned.csv"):
        fp = RESULTS / name
        if not fp.exists():
            print(f"  skip {name}: not found")
            continue
        frames.append(pd.read_csv(fp))
    if not frames:
        raise FileNotFoundError("no scored_expG_*.csv found; run the wiener/learned stages first")

    df = pd.concat(frames, ignore_index=True)
    agg = (
        df.groupby(["arm", "c"])
        .agg(
            logppl=("logppl", "mean"),
            keyword_hit=("keyword_hit", "mean"),
            sae_act=("sae_act", "mean"),
            prompt_dependence=("prompt_dependence", "mean"),
            n=("logppl", "size"),
        )
        .reset_index()
        .sort_values(["arm", "c"])
    )

    lines = [
        "# Experiment G -- arm x strength",
        "",
        "Facts pulled directly from `scored_expG_wiener.csv` and `scored_expG_learned.csv`; "
        "`naive` is present in both files and is averaged over both here.",
        "",
        "| arm | c (strength) | log-PPL | keyword_hit | sae_act | prompt_dependence | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in agg.iterrows():
        lines.append(
            f"| {row['arm']} | {row['c']:g} | {row['logppl']:.4f} | {row['keyword_hit']:.4f} | "
            f"{row['sae_act']:.4f} | {row['prompt_dependence']:.4f} | {int(row['n'])} |"
        )
    out_path = RESULTS / "expG_NOTE.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


STAGE_FUNCS = {
    "mine": stage_mine,
    "pairs_check": stage_pairs_check,
    "wiener": stage_wiener,
    "train": stage_train,
    "learned": stage_learned,
    "note": stage_note,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in requested if s not in STAGE_FUNCS]
    if unknown:
        raise SystemExit(f"unknown stage(s): {unknown}; choose from {STAGES}")

    t_all = time.time()
    failed = []
    for name in requested:
        print(f"\n=== [{name}] ===", flush=True)
        t0 = time.time()
        try:
            STAGE_FUNCS[name](args.python)
            print(f"--- [{name}] OK in {(time.time() - t0) / 60:.2f} min", flush=True)
        except Exception:
            failed.append(name)
            print(f"--- [{name}] FAILED after {(time.time() - t0) / 60:.2f} min", flush=True)
            traceback.print_exc()

    print(f"\ndone in {(time.time() - t_all) / 60:.1f} min; failed stages: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
