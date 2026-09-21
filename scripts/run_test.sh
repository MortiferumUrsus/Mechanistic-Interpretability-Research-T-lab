#!/usr/bin/env bash
# Bash mirror of run_test.ps1; the PowerShell file is the one the reported runs used.
# E4: the single TEST pass. Knobs come from configs/frozen.yaml and are not overridden here.
set -euo pipefail
PY="${PY:-python}"
cd "$(dirname "$0")/../src"

grid="0,0.5,1.0,1.25,1.5,2.0,2.5,3.0"
arms="clean,naive,norm_preserving,denoise_naive,cds,mts,fsr"

"$PY" generate.py --split test --arms "$arms" --c-grid "$grid" --n-prompts 30 --out gen_test.jsonl
"$PY" metrics.py --gen gen_test.jsonl --out scored_test.csv --split test --stages ppl,keyword,sae,dist
"$PY" pareto.py --scored scored_test.csv --concept keyword_hit
"$PY" plots.py --scored scored_test.csv --concept keyword_hit
