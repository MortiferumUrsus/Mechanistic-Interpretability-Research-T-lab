"""Exp F: new steering arms, matched-repetition / feature-distribution robustness checks, capability
vs strength, per-layer norm profile, and an entropy-penalised direction correction.

Stages, selected via --stages (comma-separated, default all in order):
    report        direction_report.py (whatever report it produces for the current direction
                  checkpoint).
    dirs          generate.py on test_r3 with the full new-arm roster (naive, dirfix, shared,
                  diffmeans, centred, purified, diffmeans_purified, rotate), then metrics.py,
                  paired_at_strength.py, control_specificity.py -> gen/scored/paired/
                  control_specificity_expF_r3.*
    entropy_comp  sweep configs/expA.yaml:kappa_shared over configs/expF.yaml:kappa_grid for the
                  `shared` arm, restoring expA.yaml afterwards; merges the per-kappa scored CSVs
                  into results/scored_expF_kappa.csv.
    qq            qq_test.py.
    matched       matched_repetition.py + feature_distribution.py on results/scored_expA_r3.csv
                  (prefixes matchedA/featdistA), and again on results/scored_expF_r3.csv if the
                  'dirs' stage has produced it (prefixes matchedF/featdistF).
    capability    capability_sweep.py.
    layers        layer_profile.py.
    entpen        train_direction_ent.py, anatomy.py --ckpt dir_ent --tag _ent (tagged outputs, so
                  dir_hot's anatomy files are untouched), then generate.py/metrics.py/
                  paired_at_strength.py for the dir_ent direction.
    note          results/expF_NOTE.md, summarising whichever of the above artefacts exist.

Each stage is wrapped in its own try/except so one failing stage does not block the others -- it
prints its own start/elapsed time, and a failure prints the traceback and moves on. Subprocesses run
from src/ with --python (default sys.executable) as the interpreter.

Three of this driver's stages ('report', 'qq', 'entpen') call scripts owned by other executors
(direction_report.py, qq_test.py, train_direction_ent.py) that may not exist on disk yet. This driver
does not write those files; instead it checks for the script's existence before running it, prints a
"missing dependency" message and skips just that stage if it is absent, so re-running this driver
later (once the dependency lands) picks the stage back up with no code change here.

    python run_expF.py [--stages report,dirs,entropy_comp,qq,matched,capability,layers,entpen,note]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"
CKPT = ROOT / "checkpoints"

STAGE_ORDER = ("report", "dirs", "entropy_comp", "qq", "matched", "capability", "layers", "entpen", "note")

DIRS_ARMS = "naive,dirfix,shared,diffmeans,centred,purified,diffmeans_purified,rotate"
DIRS_C_GRID = "0,0.5,1.0,1.5,2.0,3.0"
DEFAULT_KAPPA_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


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


def _missing(script: str) -> bool:
    """True (and prints a message) if `script` does not exist under src/ yet."""
    path = SRC / script
    if not path.exists():
        print(
            f"missing dependency: {path} does not exist yet (owned by another executor). "
            "Skipping this stage; rerun the driver once it lands."
        )
        return True
    return False


def _df_to_md(df) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for row in df.itertuples(index=False):
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def stage_report(py: str) -> None:
    if _missing("direction_report.py"):
        return
    run_cmd(py, ["direction_report.py"])


def stage_dirs(py: str) -> None:
    run_cmd(
        py,
        [
            "generate.py",
            "--split", "test_r3",
            "--arms", DIRS_ARMS,
            "--c-grid", DIRS_C_GRID,
            "--n-prompts", "30",
            "--out", "gen_expF_r3.jsonl",
        ],
    )
    run_cmd(
        py,
        [
            "metrics.py",
            "--gen", "gen_expF_r3.jsonl",
            "--out", "scored_expF_r3.csv",
            "--split", "test_r3",
            "--stages", "ppl,keyword,sae,dist",
            "--shuffle-frac", "0.15",
        ],
    )
    run_cmd(py, ["paired_at_strength.py", "--scored", "scored_expF_r3.csv", "--out", "paired_expF_r3.csv"])
    run_cmd(
        py,
        [
            "control_specificity.py",
            "--gen", "gen_expF_r3.jsonl",
            "--split", "test_r3",
            "--out", "control_specificity_expF_r3.csv",
        ],
    )


def _write_expA_kappa_shared(kappa: float) -> None:
    import yaml

    path = CONFIGS / "expA.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfg = dict(cfg or {})
    cfg["kappa_shared"] = float(kappa)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def stage_entropy_comp(py: str) -> None:
    import yaml
    import pandas as pd

    expF_path = CONFIGS / "expF.yaml"
    if not expF_path.exists():
        expF_path.write_text(
            yaml.safe_dump({"kappa_grid": DEFAULT_KAPPA_GRID}, sort_keys=False), encoding="utf-8"
        )
        print(f"configs/expF.yaml not found; wrote a default kappa_grid to {expF_path}")
    expF_cfg = yaml.safe_load(expF_path.read_text(encoding="utf-8")) or {}
    kappa_grid = [float(k) for k in expF_cfg.get("kappa_grid", DEFAULT_KAPPA_GRID)]

    expA_path = CONFIGS / "expA.yaml"
    backup = expA_path.read_bytes() if expA_path.exists() else None
    scored_paths = []
    try:
        for kappa in kappa_grid:
            _write_expA_kappa_shared(kappa)
            gen_out = f"gen_expF_kappa_{kappa}.jsonl"
            scored_out = f"scored_expF_kappa_{kappa}.csv"
            run_cmd(
                py,
                [
                    "generate.py",
                    "--split", "test_r3",
                    "--arms", "shared",
                    "--c-grid", "0,1.0,2.0",
                    "--n-prompts", "30",
                    "--tag", f"k{kappa}",
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
                    "--stages", "ppl,keyword,dist",
                    "--shuffle-frac", "0.05",
                ],
            )
            scored_paths.append((kappa, RESULTS / scored_out))
    finally:
        if backup is not None:
            expA_path.write_bytes(backup)
            print(f"restored {expA_path}")
        elif expA_path.exists():
            expA_path.unlink()
            print(f"removed {expA_path} (did not exist before this stage)")

    frames = []
    for kappa, p in scored_paths:
        if not p.exists():
            print(f"entropy_comp: {p} was not produced (generate/metrics failed for kappa={kappa}); skipping it")
            continue
        df = pd.read_csv(p)
        df["kappa"] = kappa
        frames.append(df)
    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(RESULTS / "scored_expF_kappa.csv", index=False)
        print(f"wrote results/scored_expF_kappa.csv ({len(merged)} rows, {len(frames)} kappa values)")
    else:
        print("entropy_comp: no scored files produced; nothing to merge")


def stage_qq(py: str) -> None:
    if _missing("qq_test.py"):
        return
    run_cmd(
        py,
        [
            "qq_test.py",
            "--n-docs", "500",
            "--ctx", "512",
            "--c", "1.0",
            "--features", "3",
            "--split", "test_r3",
            "--out-prefix", "qq",
        ],
    )


def stage_matched(py: str) -> None:
    scoredA = RESULTS / "scored_expA_r3.csv"
    if not scoredA.exists():
        print(f"matched: {scoredA} not found; copy it in before running this stage. Skipping.")
        return
    run_cmd(py, ["matched_repetition.py", "--scored", "scored_expA_r3.csv", "--baseline", "naive", "--out-prefix", "matchedA"])
    run_cmd(py, ["feature_distribution.py", "--scored", "scored_expA_r3.csv", "--baseline", "naive", "--c", "1.0,1.5,2.0", "--out-prefix", "featdistA"])

    scoredF = RESULTS / "scored_expF_r3.csv"
    if scoredF.exists():
        run_cmd(py, ["matched_repetition.py", "--scored", "scored_expF_r3.csv", "--baseline", "naive", "--out-prefix", "matchedF"])
        run_cmd(py, ["feature_distribution.py", "--scored", "scored_expF_r3.csv", "--baseline", "naive", "--c", "1.0,1.5,2.0", "--out-prefix", "featdistF"])
    else:
        print(f"matched: {scoredF} not found (the 'dirs' stage has not run yet); skipping the F pass")


def stage_capability(py: str) -> None:
    run_cmd(py, ["capability_sweep.py"])


def stage_layers(py: str) -> None:
    run_cmd(py, ["layer_profile.py"])


def stage_entpen(py: str) -> None:
    if _missing("train_direction_ent.py"):
        return
    run_cmd(
        py,
        [
            "train_direction_ent.py", "train",
            "--rank", "64",
            "--gamma", "1.0",
            "--lr", "3e-3",
            "--steps", "2000",
            "--name", "dir_ent",
            "--ent-weight", "1.0",
        ],
    )

    ckpt_path = CKPT / "dir_ent.pt"
    if not ckpt_path.exists():
        print(f"entpen: {ckpt_path} was not produced by train_direction_ent.py; skipping anatomy/generate for dir_ent.")
        return

    # Tagged outputs: checkpoints/shared_direction_ent.pt, results/anatomy_summary_ent.json and
    # results/anatomy_per_feature_ent.csv, so dir_hot's anatomy files are left untouched.
    run_cmd(py, ["anatomy.py", "--ckpt", "dir_ent", "--tag", "_ent"])

    run_cmd(
        py,
        [
            "generate.py",
            "--split", "test_r3",
            "--arms", "naive,dirfix",
            "--c-grid", "0,1.0,1.5,2.0,3.0",
            "--n-prompts", "30",
            "--set", "direction=dir_ent",
            "--tag", "_ent",
            "--out", "gen_expF_ent.jsonl",
        ],
    )
    run_cmd(
        py,
        [
            "metrics.py",
            "--gen", "gen_expF_ent.jsonl",
            "--out", "scored_expF_ent.csv",
            "--split", "test_r3",
            "--stages", "ppl,keyword,sae,dist",
            "--shuffle-frac", "0.15",
        ],
    )
    run_cmd(py, ["paired_at_strength.py", "--scored", "scored_expF_ent.csv", "--out", "paired_expF_ent.csv"])


def stage_note(py: str) -> None:
    import pandas as pd

    lines = ["# Exp F: new arms, robustness checks, capability, layer profile, entropy-penalised direction", ""]

    scored_path = RESULTS / "scored_expF_r3.csv"
    if scored_path.exists():
        scored = pd.read_csv(scored_path)
        table = scored.groupby(["arm", "c"])[["logppl", "keyword_hit", "rep4"]].mean().round(4).reset_index()
        lines.append("## arm x c: log-PPL, keyword_hit, rep4 (results/scored_expF_r3.csv)")
        lines.append("")
        lines.append(_df_to_md(table))
        lines.append("")
    else:
        lines.append(f"## arm x c table: {scored_path} not found; the 'dirs' stage did not complete.")
        lines.append("")

    paired_path = RESULTS / "paired_expF_r3.csv"
    if paired_path.exists():
        lines.append("## paired differences at matched strength vs naive (results/paired_expF_r3.csv)")
        lines.append("")
        lines.append(_df_to_md(pd.read_csv(paired_path).round(4)))
        lines.append("")
    else:
        lines.append(f"## paired differences: {paired_path} not found; the 'dirs' stage did not complete.")
        lines.append("")

    kappa_path = RESULTS / "scored_expF_kappa.csv"
    if kappa_path.exists():
        kdf = pd.read_csv(kappa_path)
        ktab = kdf.groupby(["kappa", "c"])[["logppl", "keyword_hit"]].mean().round(4).reset_index()
        lines.append("## kappa x c: log-PPL, keyword_hit for the `shared` arm (results/scored_expF_kappa.csv)")
        lines.append("")
        lines.append(_df_to_md(ktab))
        lines.append("")
    else:
        lines.append(f"## kappa x c table: {kappa_path} not found; the 'entropy_comp' stage did not complete.")
        lines.append("")

    for prefix, label in (("matchedA", "scored_expA_r3.csv"), ("matchedF", "scored_expF_r3.csv")):
        summary_path = RESULTS / f"{prefix}_summary.csv"
        if summary_path.exists():
            sdf = pd.read_csv(summary_path)
            overall = sdf[sdf["group"] == "all"].drop(columns=["group"])
            lines.append(f"## matched-repetition summary ({label}, results/{prefix}_summary.csv, 'all' rows)")
            lines.append("")
            lines.append(_df_to_md(overall.round(4)))
            lines.append("")
        else:
            lines.append(f"## matched-repetition summary ({label}): {summary_path} not found; the 'matched' stage did not complete.")
            lines.append("")

    cap_path = RESULTS / "capability_sweep.csv"
    if cap_path.exists():
        cdf = pd.read_csv(cap_path)
        lines.append("## capability vs strength (results/capability_sweep.csv)")
        lines.append("")
        lines.append(_df_to_md(cdf.round(4)))
        lines.append("")
    else:
        lines.append(f"## capability vs strength: {cap_path} not found; the 'capability' stage did not complete.")
        lines.append("")

    layer_path = RESULTS / "layer_profile.csv"
    if layer_path.exists():
        ldf = pd.read_csv(layer_path)
        piv = ldf.pivot_table(index=["arm", "c"], columns="layer", values="norm").reset_index()
        piv.columns = [f"layer_{c}" if isinstance(c, (int, float)) and not isinstance(c, str) else c for c in piv.columns]
        lines.append("## layer profile: mean last-position norm by layer (results/layer_profile.csv)")
        lines.append("")
        lines.append(_df_to_md(piv.round(3)))
        lines.append("")
    else:
        lines.append(f"## layer profile: {layer_path} not found; the 'layers' stage did not complete.")
        lines.append("")

    ent_summary = RESULTS / "anatomy_summary_ent.json"
    hot_summary = RESULTS / "anatomy_summary.json"
    if ent_summary.exists() and hot_summary.exists():
        ent = json.loads(ent_summary.read_text(encoding="utf-8"))
        hot = json.loads(hot_summary.read_text(encoding="utf-8"))
        lines.append("## entropy-penalised direction (dir_ent) vs the frozen direction (results/anatomy_summary_ent.json vs results/anatomy_summary.json)")
        lines.append("")
        for key in ("mean_pairwise_cos", "median_pairwise_cos", "mean_delta_norm_over_mean_norm"):
            lines.append(f"- {key}: hot={hot.get(key)}, dir_ent={ent.get(key)}")
        hot_dbar_path = CKPT / "shared_direction.pt"
        ent_dbar_path = CKPT / "shared_direction_ent.pt"
        if hot_dbar_path.exists() and ent_dbar_path.exists():
            import torch

            d_hot = torch.load(hot_dbar_path, map_location="cpu")["d_bar"].float()
            d_ent = torch.load(ent_dbar_path, map_location="cpu")["d_bar"].float()
            cos = float((d_hot @ d_ent) / (d_hot.norm() * d_ent.norm()).clamp_min(1e-9))
            norm_ratio = float(d_ent.norm() / d_hot.norm().clamp_min(1e-9))
            lines.append(f"- cos(d_bar_hot, d_bar_ent): {cos:.4f}")
            lines.append(f"- norm(d_bar_ent) / norm(d_bar_hot): {norm_ratio:.4f}")
        lines.append("")
    else:
        lines.append(f"## entpen comparison: {ent_summary} and/or {hot_summary} not found; the 'entpen' stage did not complete.")
        lines.append("")

    out_path = RESULTS / "expF_NOTE.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


STAGES = {
    "report": stage_report,
    "dirs": stage_dirs,
    "entropy_comp": stage_entropy_comp,
    "qq": stage_qq,
    "matched": stage_matched,
    "capability": stage_capability,
    "layers": stage_layers,
    "entpen": stage_entpen,
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
