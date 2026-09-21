#!/usr/bin/env bash
# Bash mirror of run_all.ps1; the PowerShell file is the one the reported runs used.
# Full reproduction pipeline. Run from the repository root.
# Every stage is idempotent and writes to data/ or results/.

set -euo pipefail
PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$0")/../src"

# E0 -- data
"$PY" activations.py verify
"$PY" activations.py dump --n-tokens 2000000 --batch-size 16
"$PY" activations.py prompts --n-prompts 40 --prompt-len 8
"$PY" sae_stats.py --chunk 4096

# E1 -- feature selection, splits, leakage table
"$PY" features.py select --seed 0

# E2 -- metric validation and frozen baselines on DEV
"$PY" generate.py --split dev --arms clean,naive,norm_preserving --out gen_dev_base.jsonl
"$PY" metrics.py --gen gen_dev_base.jsonl --out scored_dev_base.csv --split dev --stages ppl,keyword,sae,dist

# E3 -- train denoisers on FIT directions only, pick knobs on DEV.
# The budget is 60000 steps and it is not the default: the reported checkpoint is the gauss/cond1/seed0 one,
# and omitting --steps here trains 12000 instead, which is a different model from the one the report measures.
"$PY" train_denoiser.py --arch mlp --noise gauss --cond 1 --seed 0 --steps 60000
"$PY" train_denoiser.py --arch mlp --noise mix --cond 1 --seed 0 --steps 60000
"$PY" train_denoiser.py --arch linear --noise mix --cond 0 --seed 0 --steps 60000
"$PY" train_denoiser.py --arch mlp --noise mix --cond 1 --seed 1 --steps 60000

# E4 -- one pass on TEST with frozen knobs
"$PY" generate.py --split test --arms clean,naive,norm_preserving,denoise_naive,cds,mts,fsr --out gen_test.jsonl
"$PY" metrics.py --gen gen_test.jsonl --out scored_test.csv --split test --stages ppl,keyword,sae,dist
"$PY" pareto.py --scored scored_test.csv --concept keyword_hit

# E5 -- mechanism
"$PY" analysis.py transmission --split test
"$PY" analysis.py spectral --split test
"$PY" analysis.py surgery --split test
"$PY" analysis.py causal --split test --lam 1.5 --shrink 0.01
"$PY" analysis.py predictors

# Last, and it exits non-zero: every number the report quotes must be in the artefact its paragraph cites.
# Running it here is what stops a regenerated table from silently disagreeing with the text.
if ! "$PY" report_numbers_check.py; then
    echo "REPORT.md disagrees with results/; see above" >&2
    exit 1
fi

cd "$SCRIPT_DIR/.."

# E6 -- round two (direction correction) and round three (fresh feature set). Round three is a separate
# script because its holdout must be selected and excluded from training BEFORE the correction is retrained;
# see scripts/run_round3.sh, which does the selection, and README for the required order.
"$SCRIPT_DIR/run_round2.sh"
"$SCRIPT_DIR/run_round3.sh"
