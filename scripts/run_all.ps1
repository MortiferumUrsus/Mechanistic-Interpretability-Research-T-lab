# Full reproduction pipeline. Run from the repository root.
# Every stage is idempotent and writes to data/ or results/.

$ErrorActionPreference = "Stop"
$py = "$PSScriptRoot\..\.venv\Scripts\python.exe"
$src = "$PSScriptRoot\..\src"
Push-Location $src

# E0 -- data
& $py activations.py verify
& $py activations.py dump --n-tokens 2000000 --batch-size 16
& $py activations.py prompts --n-prompts 40 --prompt-len 8
& $py sae_stats.py --chunk 4096

# E1 -- feature selection, splits, leakage table
& $py features.py select --seed 0

# E2 -- metric validation and frozen baselines on DEV
& $py generate.py --split dev --arms clean,naive,norm_preserving --out gen_dev_base.jsonl
& $py metrics.py --gen gen_dev_base.jsonl --out scored_dev_base.csv --split dev --stages ppl,keyword,sae,dist

# E3 -- train denoisers on FIT directions only, pick knobs on DEV.
# The budget is 60000 steps and it is not the default: the reported checkpoint is the gauss/cond1/seed0 one,
# and omitting --steps here trains 12000 instead, which is a different model from the one the report measures.
& $py train_denoiser.py --arch mlp --noise gauss --cond 1 --seed 0 --steps 60000
& $py train_denoiser.py --arch mlp --noise mix --cond 1 --seed 0 --steps 60000
& $py train_denoiser.py --arch linear --noise mix --cond 0 --seed 0 --steps 60000
& $py train_denoiser.py --arch mlp --noise mix --cond 1 --seed 1 --steps 60000

# E4 -- one pass on TEST with frozen knobs
& $py generate.py --split test --arms clean,naive,norm_preserving,denoise_naive,cds,mts,fsr --out gen_test.jsonl
& $py metrics.py --gen gen_test.jsonl --out scored_test.csv --split test --stages ppl,keyword,sae,dist
& $py pareto.py --scored scored_test.csv --concept keyword_hit

# E5 -- mechanism
& $py analysis.py transmission --split test
& $py analysis.py spectral --split test
& $py analysis.py surgery --split test
& $py analysis.py causal --split test
& $py analysis.py predictors

Pop-Location

# E6 -- round two (direction correction) and round three (fresh feature set). Round three is a separate
# script because its holdout must be selected and excluded from training BEFORE the correction is retrained;
# see scripts/run_round3.ps1, which does the selection, and README for the required order.
& "$PSScriptRoot\run_round2.ps1"
& "$PSScriptRoot\run_round3.ps1"
