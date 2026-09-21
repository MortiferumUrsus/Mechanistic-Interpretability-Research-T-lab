#!/usr/bin/env bash
# Bash mirror of run_round2.ps1; the PowerShell file is the one the reported runs used.
# Round two: learn a direction correction, select it on DEV, evaluate once on TEST.
# Declared as a second round because the method was chosen after seeing round one.
set -euo pipefail
PY="${PY:-python}"
cd "$(dirname "$0")/../src"
grid="0,0.5,1.0,1.25,1.5,2.0,2.5,3.0"

# train the shared low-rank correction on FIT directions only
"$PY" train_direction.py train --rank 64 --gamma 1.0 --lr 3e-3 --steps 2000 --name dir_hot

# select on held-out DEV directions, in response space (no sampling, no judge)
"$PY" train_direction.py eval --checkpoints dir_hot --split dev

# text-level check on DEV, then the specificity control
"$PY" generate.py --split dev --arms naive,dirfix --c-grid "$grid" --n-prompts 10 --out gen_dev_r2.jsonl
"$PY" metrics.py --gen gen_dev_r2.jsonl --out scored_dev_r2.csv --split dev --stages ppl,keyword,sae --shuffle-frac 0.0001
"$PY" control_specificity.py --gen gen_dev_r2.jsonl --split dev --out control_specificity_dev.csv

# control: same rotation angle, random axis, same norm. If this helps as much, the gain belongs to
# perturbing the direction at all rather than to the learned correction.
"$PY" generate.py --split dev --arms randrot --c-grid 0,1.0,1.5,2.0,3.0 --n-prompts 10 --out gen_dev_ctrl.jsonl
"$PY" metrics.py --gen gen_dev_ctrl.jsonl --out scored_dev_ctrl.csv --split dev --stages ppl,keyword --shuffle-frac 0.0001
"$PY" control_specificity.py --gen gen_dev_ctrl.jsonl --split dev --out control_randrot_dev.csv

# one pass on TEST
"$PY" generate.py --split test --arms naive,dirfix --c-grid "$grid" --n-prompts 30 --out gen_test_r2.jsonl
"$PY" metrics.py --gen gen_test_r2.jsonl --out scored_test_r2.csv --split test --stages ppl,keyword,sae,dist --shuffle-frac 0.15
"$PY" pareto.py --scored scored_test_r2.csv --concept keyword_hit
"$PY" control_specificity.py --gen gen_test_r2.jsonl --split test --out control_specificity_test.csv
