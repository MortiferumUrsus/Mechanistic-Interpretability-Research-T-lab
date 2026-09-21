"""Exp E: 48 fresh features (round four) evaluated on prompts disjoint from DEV/TEST.

The headline result so far rests on 12 features (round one/three), with a correspondingly wide
interval, and DEV's prompts are a prefix of TEST's prompts, so nothing about the evaluation set
is actually held out from selection. This experiment fixes both at once: `select_round4.py` picks
48 features decorrelated from test/dev/test_r3 (with a buffer train_direction.py enforces at
training time), and a batch of 120 fresh prompts is drawn from the same held-out shard as before,
starting right after the 40 prompts DEV/TEST already use.

Resumable: each stage checks for its own output before doing the expensive part, and one stage's
failure (printed with a traceback) does not stop the others.

Two of the driver's stages ("gen" and "score") depend on CLI support another executor is adding in
parallel:
  - generate.py needs a `--prompts <path>` flag and `test_r4` as a valid `--split` choice.
  - metrics.py needs `test_r4` as a valid `--split` choice.
  - the `shared` arm used in "gen" must exist in steering.py / generate.py's `build_arms`.
This script does NOT patch those files (they belong to another executor). Instead it probes
`--help` output before running the affected commands; if the support is not there yet it prints a
clear dependency message and skips just that stage, so re-running this driver later (once the
dependency lands) picks the stage back up with no code change here.

    python run_expE.py --stages prompts,select,retrain,gen,score,note
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
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"
CKPT = ROOT / "checkpoints"

N_FEATURES = 48
SELECT_SEED = 4242
RETRAIN_NAME = "dir_r4"
N_EXISTING_PROMPTS = 40  # size of data/prompts.json today (activations.py prompts --n-prompts 40)
N_R4_PROMPTS = 120
N_PROMPTS_TOTAL = N_EXISTING_PROMPTS + N_R4_PROMPTS
STAGE_ORDER = ("prompts", "select", "retrain", "anatomy", "gen", "score", "note")


def run_cmd(py: str, args: list[str], cwd: Path = SRC) -> None:
    cmd = [py, *args]
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def cli_supports(py: str, script: str, needle: str) -> bool:
    """Best-effort probe: does `python script --help` mention `needle` (a flag or a choice)?"""
    try:
        out = subprocess.run(
            [py, script, "--help"], cwd=str(SRC), capture_output=True, text=True, check=True
        )
        return needle in out.stdout
    except Exception:
        return False


def run_stage(name: str, fn) -> None:
    print(f"\n=== stage '{name}' start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    t0 = time.time()
    try:
        fn()
        print(f"=== stage '{name}' done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    except Exception:
        print(f"!!! stage '{name}' FAILED after {(time.time() - t0) / 60:.1f} min !!!", flush=True)
        traceback.print_exc()


def stage_prompts(py: str) -> None:
    """Get 120 fresh prompts after the first 40, without overwriting data/prompts.json.

    activations.py's `prompts` stage always writes to a hardcoded data/prompts.json and has no
    output-path flag; it belongs to another executor, so it is not patched here. Instead:
    back up prompts.json's current bytes in memory, ask for N_PROMPTS_TOTAL prompts (which,
    since selection is a deterministic walk over a fixed corpus shard, reproduces the existing
    first N_EXISTING_PROMPTS verbatim and then continues past them), slice off the new tail into
    data/prompts_r4.json, and restore the original prompts.json from the in-memory backup in
    `finally` regardless of outcome.
    """
    prompts_path = DATA / "prompts.json"
    backup = prompts_path.read_bytes() if prompts_path.exists() else None
    try:
        run_cmd(
            py,
            ["activations.py", "prompts", "--n-prompts", str(N_PROMPTS_TOTAL), "--prompt-len", "8"],
        )
        new_prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        if backup is not None:
            old_prompts = json.loads(backup.decode("utf-8"))
            if new_prompts[: len(old_prompts)] != old_prompts:
                raise RuntimeError(
                    "the first "
                    f"{len(old_prompts)} prompts of the fresh {N_PROMPTS_TOTAL}-prompt draw do not "
                    "match the existing data/prompts.json prefix; activations.py's prompt selection "
                    "is no longer deterministic against the same seed/corpus, or n-prompts/prompt-len "
                    "changed underneath this script. Refusing to guess which prompts are actually "
                    "fresh."
                )
            start = len(old_prompts)
        else:
            start = N_EXISTING_PROMPTS
        r4_prompts = new_prompts[start:]
        if len(r4_prompts) != N_R4_PROMPTS:
            raise RuntimeError(
                f"expected {N_R4_PROMPTS} fresh prompts after the first {start}, got {len(r4_prompts)} "
                f"(requested {N_PROMPTS_TOTAL} total; the corpus shard may have run out early)"
            )
        out_path = DATA / "prompts_r4.json"
        out_path.write_text(json.dumps(r4_prompts, indent=2), encoding="utf-8")
        print(f"wrote {len(r4_prompts)} fresh prompts to {out_path} (prompts {start}..{start + len(r4_prompts) - 1})")
    finally:
        if backup is not None:
            prompts_path.write_bytes(backup)
            print(f"restored {prompts_path} to its pre-existing content")
        elif prompts_path.exists():
            prompts_path.unlink()
            print(f"removed {prompts_path} (did not exist before this stage)")


def stage_select(py: str) -> None:
    if "test_r4" in _load_features_yaml():
        print("select: configs/features_r4.yaml already defines test_r4; skipping (run select_round4.py --overwrite manually if a re-select is wanted)")
        return
    run_cmd(py, ["select_round4.py", "--n", str(N_FEATURES), "--seed", str(SELECT_SEED)])


def stage_retrain(py: str) -> None:
    ckpt_path = CKPT / f"{RETRAIN_NAME}.pt"
    if ckpt_path.exists():
        print(f"retrain: {ckpt_path} already exists, skipping")
        return
    run_cmd(
        py,
        [
            "train_direction.py", "train",
            "--rank", "64",
            "--gamma", "1.0",
            "--lr", "3e-3",
            "--steps", "2000",
            "--name", RETRAIN_NAME,
            "--exclude-r4",
        ],
    )


def stage_anatomy(py: str) -> None:
    """Derive the shared direction from the ROUND-FOUR checkpoint, not from dir_hot.

    The `shared` arm injects `v + kappa * d_bar`, and d_bar is an average over the correction the
    map applies to fit directions. Taking it from dir_hot would import a checkpoint trained back
    when the forty-eight round-four features were still in the pool -- a weak leak, but the same
    shape of leak that invalidated the first version of round three. dir_r4 excludes them and
    their cosine neighbours, so d_bar derived from it has never seen this split.
    """
    ckpt_path = CKPT / f"{RETRAIN_NAME}.pt"
    if not ckpt_path.exists():
        print(f"anatomy: {ckpt_path} not found; run the 'retrain' stage first. Skipping.")
        return
    run_cmd(py, ["anatomy.py", "--ckpt", RETRAIN_NAME])


def stage_gen(py: str) -> None:
    prompts_path = DATA / "prompts_r4.json"
    if not prompts_path.exists():
        print(f"gen: {prompts_path} not found; run the 'prompts' stage first. Skipping.")
        return
    if not (CKPT / f"{RETRAIN_NAME}.pt").exists():
        print(f"gen: checkpoint {RETRAIN_NAME}.pt not found; run the 'retrain' stage first. Skipping.")
        return
    has_prompts_flag = cli_supports(py, "generate.py", "--prompts")
    has_split_r4 = cli_supports(py, "generate.py", "test_r4")
    if not (has_prompts_flag and has_split_r4):
        print(
            "gen: generate.py does not yet support --prompts and/or split=test_r4 "
            "(dependency: another executor is adding these). Skipping; rerun this driver once "
            "that lands. (has --prompts: "
            f"{has_prompts_flag}, has test_r4 split choice: {has_split_r4})"
        )
        return
    run_cmd(
        py,
        [
            "generate.py",
            "--split", "test_r4",
            "--arms", "naive,dirfix,shared",
            "--c-grid", "0,0.5,1.0,1.5,2.0,3.0",
            "--n-prompts", "60",
            "--prompts", str(prompts_path),
            "--set", f"direction={RETRAIN_NAME}",
            "--out", "gen_expE_r4.jsonl",
        ],
    )


def stage_score(py: str) -> None:
    gen_path = RESULTS / "gen_expE_r4.jsonl"
    if not gen_path.exists():
        print(f"score: {gen_path} not found; the gen stage did not produce it yet. Skipping.")
        return
    has_split_r4 = cli_supports(py, "metrics.py", "test_r4")
    if not has_split_r4:
        print(
            "score: metrics.py does not yet accept split=test_r4 (dependency: another executor is "
            "adding this). Skipping; rerun this driver once that lands."
        )
        return
    run_cmd(
        py,
        [
            "metrics.py",
            "--gen", "gen_expE_r4.jsonl",
            "--out", "scored_expE_r4.csv",
            "--split", "test_r4",
            "--stages", "ppl,keyword,sae,dist",
            "--shuffle-frac", "0.1",
        ],
    )
    run_cmd(py, ["pareto.py", "--scored", "scored_expE_r4.csv", "--concept", "keyword_hit", "--prefix", "r4_"])
    run_cmd(py, ["paired_at_strength.py", "--scored", "scored_expE_r4.csv", "--out", "paired_expE_r4.csv"])


def _load_features_yaml() -> dict:
    import yaml

    path = CONFIGS / "features_r4.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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

    lines = ["# Exp E: 48 fresh features on disjoint prompts (round four)", ""]

    payload = _load_features_yaml()
    n_features = len(payload["test_r4"]) if "test_r4" in payload else None
    prompts_r4_path = DATA / "prompts_r4.json"
    n_prompts = (
        len(json.loads(prompts_r4_path.read_text(encoding="utf-8"))) if prompts_r4_path.exists() else None
    )
    lines.append(
        f"- features in `configs/features_r4.yaml:test_r4`: "
        f"{n_features if n_features is not None else 'not selected yet (select stage did not run)'}"
    )
    lines.append(
        f"- fresh prompts in `data/prompts_r4.json`: "
        f"{n_prompts if n_prompts is not None else 'not generated yet (prompts stage did not run)'}"
    )
    lines.append("")

    summary_path = RESULTS / "r4_endpoint_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        lines.append("## primary endpoint (concept_at_budget, budget = naive at c=1.0), test_r4")
        lines.append(_df_to_md(df))
        lines.append("")
    else:
        lines.append(f"## endpoint: {summary_path} not found; the score stage did not complete.")
        lines.append("")

    paired_path = RESULTS / "paired_expE_r4.csv"
    if paired_path.exists():
        df = pd.read_csv(paired_path)
        lines.append("## paired differences at matched strength vs naive, test_r4")
        lines.append(_df_to_md(df))
    else:
        lines.append(f"## paired differences: {paired_path} not found; the score stage did not complete.")

    out_path = RESULTS / "expE_NOTE.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


STAGES = {
    "prompts": stage_prompts,
    "select": stage_select,
    "retrain": stage_retrain,
    "anatomy": stage_anatomy,
    "gen": stage_gen,
    "score": stage_score,
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
