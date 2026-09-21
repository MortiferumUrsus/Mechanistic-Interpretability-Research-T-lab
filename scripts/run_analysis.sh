#!/usr/bin/env bash
# Bash mirror of run_analysis.ps1; the PowerShell file is the one the reported runs used.
# E5: mechanism measurements. Runs after the TEST pass; reads frozen.yaml for the arm settings.
# The PowerShell original sets $ErrorActionPreference = "Continue", so this mirror does not use `set -e`:
# a failing stage is reported and the rest still run.
set -uo pipefail
PY="${PY:-python}"
cd "$(dirname "$0")/../src"

# how much of the steering each denoiser transmits, learned and closed-form side by side
"$PY" analysis.py transmission --split test --denoisers mlp_mix_cond1_s0,mlp_gauss_cond1_s0,mlp_fixed200,wiener:0.5,wiener:1.0,wiener:2.0

# where the perturbation energy sits in the covariance eigenbasis, before and after each repair
"$PY" analysis.py spectral --split test

# which non-target latents get clamped, and how many
"$PY" analysis.py surgery --split test

# tangent-aligned versus off-tangent downstream response
"$PY" analysis.py causal --split test --lam 1.5 --shrink 0.01

# how far generation-time activations drift from the training corpus
"$PY" analysis.py rollout_shift --split test --max-features 4

# per-feature gain against the two registered geometric predictors
"$PY" analysis.py predictors

# Derived tables the report cites, so section 9.1 has an artefact rather than arithmetic in prose.
"$PY" dirfix_vs_naive.py

"$PY" plots.py --scored scored_test.csv --concept keyword_hit
"$PY" make_html.py

# Last, and it exits non-zero: every number the report quotes must be in the artefact its paragraph cites.
# Running it here is what stops a regenerated table from silently disagreeing with the text.
"$PY" report_numbers_check.py || echo "REPORT.md disagrees with results/; see above" >&2
