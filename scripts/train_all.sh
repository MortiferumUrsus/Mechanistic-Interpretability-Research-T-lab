#!/usr/bin/env bash
# Bash mirror of train_all.ps1; the PowerShell file is the one the reported runs used.
# Train every denoiser variant used in the ablations. ~6 minutes each on a GTX 1660 Ti.
# The PowerShell original sets $ErrorActionPreference = "Continue", so this mirror does not use `set -e`.
set -uo pipefail
PY="${PY:-python}"
cd "$(dirname "$0")/../src"
steps=60000

"$PY" train_denoiser.py --arch mlp    --noise mix   --cond 1 --seed 0 --steps "$steps" --log-every 20000
"$PY" train_denoiser.py --arch mlp    --noise gauss --cond 1 --seed 0 --steps "$steps" --log-every 20000
"$PY" train_denoiser.py --arch linear --noise mix   --cond 0 --seed 0 --steps "$steps" --log-every 20000
"$PY" train_denoiser.py --arch mlp    --noise mix   --cond 0 --seed 0 --steps "$steps" --log-every 20000
"$PY" train_denoiser.py --arch mlp    --noise mix   --cond 1 --seed 1 --steps "$steps" --log-every 20000
"$PY" train_denoiser.py --arch mlp    --noise mix   --cond 1 --seed 0 --steps "$steps" --log-every 20000 --fixed-rel 2.0 --name mlp_fixed200
