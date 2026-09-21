"""Exp D: how much of the direction-correction result is seed/rank overfitting.

Round two (`dir_hot`, rank 64) and round three both report a single checkpoint at a single seed.
If the correction is fragile to its own training randomness, that single number is not a claim
about the method, it is a claim about one lucky run. This script trains a rank x seed grid,
scores it cheaply on DEV, and re-runs the round-three TEST comparison (naive vs dirfix) at the
rank actually used (64) across all five seeds, so the round-three paired numbers get a spread
instead of a point estimate.

Resumable: every subprocess call is idempotent-checked against files already on disk (a trained
checkpoint, an existing scored/paired csv) before it runs, so re-invoking this script after a
partial run, a crash, or a Kaggle timeout only does the remaining work. Each stage is independent:
one stage's failure is printed (with a traceback) and does not stop the others.

    python run_expD.py --stages train,eval_dev,gen_r3,note
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CKPT = ROOT / "checkpoints"

RANKS = (8, 64, 256)
SEEDS = (0, 1, 2, 3, 4)
GEN_RANK = 64
STAGE_ORDER = ("train", "eval_dev", "gen_r3", "note")


def ckpt_name(rank: int, seed: int) -> str:
    return f"dir_r{rank}_s{seed}"


def run_cmd(py: str, args: list[str], cwd: Path = SRC) -> None:
    cmd = [py, *args]
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_stage(name: str, fn) -> None:
    print(f"\n=== stage '{name}' start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    t0 = time.time()
    try:
        fn()
        print(f"=== stage '{name}' done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    except Exception:
        print(f"!!! stage '{name}' FAILED after {(time.time() - t0) / 60:.1f} min !!!", flush=True)
        traceback.print_exc()


def stage_train(py: str) -> None:
    for rank in RANKS:
        for seed in SEEDS:
            name = ckpt_name(rank, seed)
            ckpt_path = CKPT / f"{name}.pt"
            if ckpt_path.exists():
                print(f"skip {name}: {ckpt_path} already exists")
                continue
            try:
                run_cmd(
                    py,
                    [
                        "train_direction.py", "train",
                        "--rank", str(rank),
                        "--gamma", "1.0",
                        "--lr", "3e-3",
                        "--steps", "2000",
                        "--seed", str(seed),
                        "--name", name,
                    ],
                )
            except Exception:
                print(f"train: {name} failed", flush=True)
                traceback.print_exc()


def stage_eval_dev(py: str) -> None:
    """Reproduce the cheap no-generation eval that produces results/dirfix_eval_dev.csv
    (`train_direction.py eval --checkpoints <names> --split dev`, see train_direction.py's
    `evaluate()`), for every rank/seed checkpoint, without touching the existing
    results/dirfix_eval_dev.csv file left over from round two.
    """
    import pandas as pd

    names = [ckpt_name(r, s) for r in RANKS for s in SEEDS]
    existing = [n for n in names if (CKPT / f"{n}.pt").exists()]
    missing = [n for n in names if n not in existing]
    if missing:
        print(f"eval_dev: {len(missing)} checkpoint(s) not trained yet, skipping them: {missing}")
    if not existing:
        print("eval_dev: no expD checkpoints found on disk; nothing to evaluate")
        return

    target = RESULTS / "dirfix_eval_dev.csv"
    backup = target.read_bytes() if target.exists() else None
    try:
        run_cmd(py, ["train_direction.py", "eval", "--checkpoints", ",".join(existing), "--split", "dev"])
        if not target.exists():
            raise RuntimeError(f"expected {target} to be written by train_direction.py eval")
        df = pd.read_csv(target)
        parsed = df["checkpoint"].str.extract(r"^dir_r(?P<rank>\d+)_s(?P<seed>\d+)$")
        df["rank"] = parsed["rank"].astype(int)
        df["seed"] = parsed["seed"].astype(int)
        agg = (
            df.groupby(["rank", "seed", "c"])[["a_ratio", "c_ratio"]]
            .mean()
            .reset_index()
            .rename(columns={"a_ratio": "A_ratio", "c_ratio": "C_ratio"})
            .sort_values(["rank", "seed", "c"])
        )
        out_path = RESULTS / "expD_eval_dev.csv"
        agg.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(agg)} rows)")
        print(agg.to_string(index=False))
    finally:
        # This command always overwrites results/dirfix_eval_dev.csv; restore whatever was there
        # before (round two's own eval), since existing results/ files are not ours to change.
        if backup is not None:
            target.write_bytes(backup)
            print(f"restored {target} to its pre-existing content")
        elif target.exists():
            target.unlink()
            print(f"removed {target} (did not exist before this stage)")


def stage_gen_r3(py: str) -> None:
    """rank=64 only, all five seeds: repeat the round-three naive-vs-dirfix comparison
    (scripts/run_round3.ps1's generate/metrics/paired_at_strength trio) once per seed, swapping in
    that seed's checkpoint via `--set direction=...` (see generate.py: `--set KEY=VALUE` overrides
    a key of the frozen hyper-parameter dict, and `dirfix` reads `frozen['direction']`).
    """
    for seed in SEEDS:
        name = ckpt_name(GEN_RANK, seed)
        if not (CKPT / f"{name}.pt").exists():
            print(f"gen_r3: checkpoint {name} missing, skipping seed {seed}")
            continue
        gen_out = f"gen_expD_s{seed}.jsonl"
        scored_out = f"scored_expD_s{seed}.csv"
        paired_out = f"paired_expD_s{seed}.csv"
        try:
            run_cmd(
                py,
                [
                    "generate.py",
                    "--split", "test_r3",
                    "--arms", "naive,dirfix",
                    "--c-grid", "0,1.0,1.5,2.0,3.0",
                    "--n-prompts", "30",
                    "--set", f"direction={name}",
                    "--out", gen_out,
                ],
            )
            run_cmd(
                py,
                [
                    "metrics.py",
                    "--gen", gen_out,
                    "--out", scored_out,
                    "--split", "test_r3",
                    "--stages", "ppl,keyword",
                    "--shuffle-frac", "0.05",
                ],
            )
            run_cmd(py, ["paired_at_strength.py", "--scored", scored_out, "--out", paired_out])
        except Exception:
            print(f"gen_r3: seed {seed} failed", flush=True)
            traceback.print_exc()


def _df_to_md(df) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for row in df.itertuples(index=False):
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def stage_note(py: str) -> None:
    import pandas as pd

    lines = ["# Exp D: seed / rank sensitivity of the direction correction", ""]

    eval_path = RESULTS / "expD_eval_dev.csv"
    if eval_path.exists():
        df = pd.read_csv(eval_path)
        pivot = df.groupby(["rank", "seed"])[["A_ratio", "C_ratio"]].mean().reset_index()
        lines.append("## rank x seed, DEV (mean over c and dev features, from expD_eval_dev.csv)")
        lines.append(_df_to_md(pivot))
        lines.append("")
    else:
        lines.append(f"## rank x seed: {eval_path} not found; the eval_dev stage did not complete.")
        lines.append("")

    paired_files = sorted(RESULTS.glob("paired_expD_s*.csv"))
    if paired_files:
        rows = []
        for p in paired_files:
            seed = int(p.stem.rsplit("_s", 1)[1])
            d = pd.read_csv(p)
            d = d[d["arm"] == "dirfix"]
            if d.empty:
                continue
            r = d.iloc[0]
            rows.append(
                {
                    "seed": seed,
                    "d_logppl": r["d_logppl"],
                    "logppl_lo95": r["logppl_lo95"],
                    "logppl_hi95": r["logppl_hi95"],
                    "d_keyword_hit": r["d_concept"],
                    "keyword_lo95": r["concept_lo95"],
                    "keyword_hi95": r["concept_hi95"],
                }
            )
        if rows:
            t = pd.DataFrame(rows).sort_values("seed")
            lines.append(f"## paired dirfix - naive at matched strength, rank={GEN_RANK}, test_r3, per seed")
            lines.append(_df_to_md(t))
            lines.append("")
            summ = pd.DataFrame(
                {
                    "metric": ["d_logppl", "d_keyword_hit"],
                    "mean": [t["d_logppl"].mean(), t["d_keyword_hit"].mean()],
                    "min": [t["d_logppl"].min(), t["d_keyword_hit"].min()],
                    "max": [t["d_logppl"].max(), t["d_keyword_hit"].max()],
                }
            )
            lines.append("## across the five seeds")
            lines.append(_df_to_md(summ))
        else:
            lines.append("## paired differences: no 'dirfix' rows found in paired_expD_s*.csv")
    else:
        lines.append("## paired differences: no results/paired_expD_s*.csv found; the gen_r3 stage did not complete.")

    out_path = RESULTS / "expD_NOTE.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


STAGES = {
    "train": stage_train,
    "eval_dev": stage_eval_dev,
    "gen_r3": stage_gen_r3,
    "note": stage_note,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", default=sys.executable, help="interpreter for subprocess calls")
    ap.add_argument(
        "--stages",
        default=",".join(STAGE_ORDER),
        help=f"comma-separated subset of: {','.join(STAGE_ORDER)}",
    )
    args = ap.parse_args()

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in requested if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")

    for stage in requested:
        run_stage(stage, lambda s=stage: STAGES[s](args.python))


if __name__ == "__main__":
    main()
