"""Exp A: anatomy of the round-two direction correction, and the shared/residual/antimanifold arms.

Round two's correction is a shared low-rank map w(v) = normalize(v + M v). `anatomy.py` asks how
coherent that map's effect is across features -- whether it mostly pushes every direction the same
way -- and exposes that shared push as its own steering arm (`shared`), a version of the correction
with the shared push removed (`residual`), and an unrelated control that leans the other way in the
activation covariance (`antimanifold`, plus its direction-free control `shared_only`).

Stages, selected via --stages (comma-separated, default all in order):
    anatomy  run anatomy.py: writes checkpoints/shared_direction.pt, results/anatomy_summary.json,
             results/anatomy_per_feature.csv.
    sweep    pick kappa_shared / kappa_anti on DEV against a naive reference, write the choice to
             configs/expA.yaml and the full grid to results/expA_sweep.csv.
    r3       generate + score the round-three comparison (naive, dirfix, shared, residual,
             antimanifold, shared_only) and its paired/specificity tables.
    note     write results/expA_NOTE.md from the artefacts the earlier stages produced.

Each stage is wrapped in its own try/except so one failing stage does not block the others -- it
prints its own start/elapsed time, and a failure prints the traceback and moves on. Subprocesses run
from src/ with --python (default sys.executable) as the interpreter. Only pathlib, no literal
backslashes, so this also runs on Linux (Kaggle).

    python run_expA.py [--stages anatomy,sweep,r3,note] [--python /path/to/python]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"

KAPPAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SWEEP_ARMS = ("shared", "antimanifold")
KAPPA_KEY = {"shared": "kappa_shared", "antimanifold": "kappa_anti"}
SWEEP_STRENGTHS = (1.0, 2.0)
STAGE_ORDER = ("anatomy", "sweep", "r3", "note")


def run_cmd(py: str, args: list[str]) -> None:
    cmd = [py, *args]
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(SRC), check=True)


def run_stage(name: str, fn) -> None:
    print(f"\n=== stage '{name}' start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    t0 = time.time()
    try:
        fn()
        print(f"=== stage '{name}' done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    except Exception:
        print(f"!!! stage '{name}' FAILED after {(time.time() - t0) / 60:.1f} min !!!", flush=True)
        traceback.print_exc()


def write_expA_yaml(updates: dict) -> None:
    import yaml

    path = CONFIGS / "expA.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfg = dict(cfg or {})
    cfg.update(updates)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _df_to_md(df) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for row in df.itertuples(index=False):
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def stage_anatomy(py: str) -> None:
    run_cmd(py, ["anatomy.py"])


def stage_sweep(py: str) -> None:
    import pandas as pd

    # reference run, once: naive on the same DEV grid the swept arms use.
    run_cmd(
        py,
        [
            "generate.py",
            "--split", "dev",
            "--arms", "naive",
            "--c-grid", "1.0,2.0",
            "--n-prompts", "10",
            "--out", "dev_expA_naive.jsonl",
        ],
    )
    run_cmd(
        py,
        [
            "metrics.py",
            "--gen", "dev_expA_naive.jsonl",
            "--out", "scored_expA_naive.csv",
            "--split", "dev",
            "--stages", "ppl,keyword",
            "--shuffle-frac", "0.05",
        ],
    )
    naive_by_c = pd.read_csv(RESULTS / "scored_expA_naive.csv").groupby("c")[["logppl", "keyword_hit"]].mean()

    rows = []
    for arm in SWEEP_ARMS:
        for kappa in KAPPAS:
            write_expA_yaml({KAPPA_KEY[arm]: kappa})
            gen_out = f"dev_expA_{arm}_k{kappa}.jsonl"
            scored_out = f"scored_expA_{arm}_k{kappa}.csv"
            run_cmd(
                py,
                [
                    "generate.py",
                    "--split", "dev",
                    "--arms", arm,
                    "--c-grid", "1.0,2.0",
                    "--n-prompts", "10",
                    "--out", gen_out,
                ],
            )
            run_cmd(
                py,
                [
                    "metrics.py",
                    "--gen", gen_out,
                    "--out", scored_out,
                    "--split", "dev",
                    "--stages", "ppl,keyword",
                    "--shuffle-frac", "0.05",
                ],
            )
            by_c = pd.read_csv(RESULTS / scored_out).groupby("c")[["logppl", "keyword_hit"]].mean()
            for c in SWEEP_STRENGTHS:
                rows.append(
                    {
                        "arm": arm,
                        "kappa": kappa,
                        "c": c,
                        "mean_logppl": float(by_c.loc[c, "logppl"]),
                        "mean_keyword_hit": float(by_c.loc[c, "keyword_hit"]),
                        "naive_mean_logppl": float(naive_by_c.loc[c, "logppl"]),
                        "naive_mean_keyword_hit": float(naive_by_c.loc[c, "keyword_hit"]),
                    }
                )
    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(RESULTS / "expA_sweep.csv", index=False)
    print(f"wrote results/expA_sweep.csv ({len(sweep_df)} rows)")

    # Deterministic selection, per arm: among kappa whose mean log-PPL is not above naive's AND
    # whose mean keyword_hit is not below naive's, at BOTH strengths, take the one with the highest
    # keyword_hit (averaged over both strengths). If none qualify, drop the keyword_hit requirement
    # and take the highest keyword_hit among those that still do not cost log-PPL. If even that is
    # empty, fall back to kappa=1.0.
    chosen = {}
    for arm in SWEEP_ARMS:
        sub = sweep_df[sweep_df["arm"] == arm]
        strict_ok, loose_ok = [], []
        for kappa in KAPPAS:
            g = sub[sub["kappa"] == kappa]
            ppl_ok = bool((g["mean_logppl"] <= g["naive_mean_logppl"]).all())
            kw_ok = bool((g["mean_keyword_hit"] >= g["naive_mean_keyword_hit"]).all())
            mean_kw = float(g["mean_keyword_hit"].mean())
            if ppl_ok and kw_ok:
                strict_ok.append((kappa, mean_kw))
            if ppl_ok:
                loose_ok.append((kappa, mean_kw))
        if strict_ok:
            best = max(strict_ok, key=lambda t: t[1])[0]
        elif loose_ok:
            best = max(loose_ok, key=lambda t: t[1])[0]
        else:
            best = 1.0
        chosen[KAPPA_KEY[arm]] = float(best)
        print(f"selected {KAPPA_KEY[arm]} = {best} for arm '{arm}'")

    write_expA_yaml(chosen)
    print(f"wrote {chosen} to configs/expA.yaml")


def stage_r3(py: str) -> None:
    run_cmd(
        py,
        [
            "generate.py",
            "--split", "test_r3",
            "--arms", "naive,dirfix,shared,residual,antimanifold,shared_only",
            "--c-grid", "0,0.5,1.0,1.5,2.0,3.0,4.0,5.0",
            "--n-prompts", "30",
            "--out", "gen_expA_r3.jsonl",
        ],
    )
    run_cmd(
        py,
        [
            "metrics.py",
            "--gen", "gen_expA_r3.jsonl",
            "--out", "scored_expA_r3.csv",
            "--split", "test_r3",
            "--stages", "ppl,keyword,sae,dist",
            "--shuffle-frac", "0.15",
        ],
    )
    run_cmd(py, ["paired_at_strength.py", "--scored", "scored_expA_r3.csv", "--out", "paired_expA_r3.csv"])
    run_cmd(
        py,
        [
            "control_specificity.py",
            "--gen", "gen_expA_r3.jsonl",
            "--split", "test_r3",
            "--out", "control_specificity_expA_r3.csv",
        ],
    )


def stage_note(py: str) -> None:
    import pandas as pd
    import yaml

    lines = ["# Exp A: anatomy of the direction correction, and the shared/residual/antimanifold arms", ""]

    expA_path = CONFIGS / "expA.yaml"
    if expA_path.exists():
        cfg = yaml.safe_load(expA_path.read_text(encoding="utf-8")) or {}
        lines.append("## Selected kappa (configs/expA.yaml)")
        lines.append("")
        lines.append(f"- kappa_shared: {cfg.get('kappa_shared')}")
        lines.append(f"- kappa_anti: {cfg.get('kappa_anti')}")
        lines.append("")
    else:
        lines.append(f"## Selected kappa: {expA_path} not found; the sweep stage did not complete.")
        lines.append("")

    summary_path = RESULTS / "anatomy_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lines.append("## Anatomy of the correction (results/anatomy_summary.json)")
        lines.append("")
        for k, v in summary.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    else:
        lines.append(f"## Anatomy: {summary_path} not found; the anatomy stage did not complete.")
        lines.append("")

    scored_path = RESULTS / "scored_expA_r3.csv"
    if scored_path.exists():
        scored = pd.read_csv(scored_path)
        table = scored.groupby(["arm", "c"])[["logppl", "keyword_hit"]].mean().round(4).reset_index()
        lines.append("## Arm x strength: log-PPL and keyword_hit, test_r3 (results/scored_expA_r3.csv)")
        lines.append("")
        lines.append(_df_to_md(table))
        lines.append("")
    else:
        lines.append(f"## Arm x strength table: {scored_path} not found; the r3 stage did not complete.")
        lines.append("")

    out_path = RESULTS / "expA_NOTE.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


STAGES = {
    "anatomy": stage_anatomy,
    "sweep": stage_sweep,
    "r3": stage_r3,
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
