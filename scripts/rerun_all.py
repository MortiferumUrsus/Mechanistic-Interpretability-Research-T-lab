"""Re-run every generation that the direction-cache defect touched, on the retrained correction.

Two things changed under these numbers and both invalidate every earlier generation of the two arms that
use a cached direction:

- `CorrectedDirectionArm` and `RandomRotationArm` keyed their cache on `id(v_hat)` while holding no
  reference to the tensor, so a later feature could be steered with an earlier feature's direction;
- round three's features are now held out of the correction's training pool, so `dir_hot` itself is a
  different (and honestly held-out) checkpoint.

Every `dirfix` and `randrot` artefact therefore has to be regenerated, and the `naive` arm with them --
not because naive was affected, but because the paired statistics compare the two arms within one file.

Order matters: dev first, so the sanity checks fail cheaply before the thirty-prompt TEST passes start.

    python scripts/rerun_all.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
PY = str(Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe")
GRID = "0,0.5,1.0,1.25,1.5,2.0,2.5,3.0"

STEPS: list[list[str]] = [
    # The checkpoint's own evaluation first: if the retrained correction does not clear its dev numbers,
    # nothing below is worth running.
    ["train_direction.py", "eval", "--checkpoints", "dir_hot", "--split", "dev"],

    # --- DEV: cheap, and the place where a broken pipeline should surface.
    ["generate.py", "--split", "dev", "--arms", "naive,dirfix", "--c-grid", GRID,
     "--n-prompts", "10", "--out", "gen_dev_r2.jsonl"],
    ["metrics.py", "--gen", "gen_dev_r2.jsonl", "--out", "scored_dev_r2.csv", "--split", "dev",
     "--stages", "ppl,keyword,sae", "--shuffle-frac", "0.0001"],
    ["control_specificity.py", "--gen", "gen_dev_r2.jsonl", "--split", "dev",
     "--out", "control_specificity_dev.csv"],
    ["generate.py", "--split", "dev", "--arms", "randrot", "--c-grid", "0,1.0,1.5,2.0,3.0",
     "--n-prompts", "10", "--out", "gen_dev_ctrl.jsonl"],
    ["metrics.py", "--gen", "gen_dev_ctrl.jsonl", "--out", "scored_dev_ctrl.csv", "--split", "dev",
     "--stages", "ppl,keyword", "--shuffle-frac", "0.0001"],
    ["control_specificity.py", "--gen", "gen_dev_ctrl.jsonl", "--split", "dev",
     "--out", "control_randrot_dev.csv"],

    # --- TEST, round two.
    ["generate.py", "--split", "test", "--arms", "naive,dirfix", "--c-grid", GRID,
     "--n-prompts", "30", "--out", "gen_test_r2.jsonl"],
    ["metrics.py", "--gen", "gen_test_r2.jsonl", "--out", "scored_test_r2.csv", "--split", "test",
     "--stages", "ppl,keyword,sae,dist", "--shuffle-frac", "0.15"],
    ["pareto.py", "--scored", "scored_test_r2.csv", "--concept", "keyword_hit"],
    ["paired_at_strength.py", "--scored", "scored_test_r2.csv", "--out", "paired_test_r2.csv"],
    ["control_specificity.py", "--gen", "gen_test_r2.jsonl", "--split", "test",
     "--out", "control_specificity_test.csv"],
    ["matched_coordinate.py", "--scored", "scored_test_r2.csv"],

    # --- TEST, round three, on features the retrained correction has never seen.
    ["generate.py", "--split", "test_r3", "--arms", "naive,dirfix", "--c-grid", GRID,
     "--n-prompts", "30", "--out", "gen_r3.jsonl"],
    ["metrics.py", "--gen", "gen_r3.jsonl", "--out", "scored_r3.csv", "--split", "test_r3",
     "--stages", "ppl,keyword,sae,dist", "--shuffle-frac", "0.15"],
    ["pareto.py", "--scored", "scored_r3.csv", "--concept", "keyword_hit", "--prefix", "r3_"],
    ["paired_at_strength.py", "--scored", "scored_r3.csv", "--out", "paired_r3.csv"],
    ["control_specificity.py", "--gen", "gen_r3.jsonl", "--split", "test_r3",
     "--out", "control_specificity_r3.csv"],
    ["matched_coordinate.py", "--scored", "scored_r3.csv", "--prefix", "r3_"],

    # --- the rotation control on a real test split.
    ["generate.py", "--split", "test_r3", "--arms", "randrot", "--c-grid", "0,1.0,1.5,2.0,3.0",
     "--n-prompts", "30", "--out", "gen_r3_ctrl.jsonl"],
    ["metrics.py", "--gen", "gen_r3_ctrl.jsonl", "--out", "scored_r3_ctrl.csv", "--split", "test_r3",
     "--stages", "ppl,keyword", "--shuffle-frac", "0.0001"],
    ["control_specificity.py", "--gen", "gen_r3_ctrl.jsonl", "--split", "test_r3",
     "--out", "control_randrot_r3.csv"],
]


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8")

    t0 = time.time()
    for i, step in enumerate(STEPS, 1):
        print(f"\n=== [{i}/{len(STEPS)}] {' '.join(step)}", flush=True)
        r = subprocess.run([PY] + step, cwd=SRC)
        if r.returncode:
            # Stop rather than continue: every later step reads what an earlier one wrote, so carrying on
            # after a failure produces a mixture of old and new artefacts -- which is the exact state this
            # whole re-run exists to clear.
            print(f"FAILED at step {i} ({step[0]}) with code {r.returncode}; stopping", flush=True)
            raise SystemExit(r.returncode)
    print(f"\nall {len(STEPS)} steps done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
