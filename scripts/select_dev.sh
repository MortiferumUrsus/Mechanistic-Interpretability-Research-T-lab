#!/usr/bin/env bash
# Bash mirror of select_dev.ps1; the PowerShell file is the one the reported runs used.
# DEV knob selection. Every candidate is judged by the same metric the report uses, on DEV
# features only. TEST is never generated here.
# The PowerShell original sets $ErrorActionPreference = "Continue", so this mirror does not use `set -e`.
set -uo pipefail
PY="${PY:-python}"
cd "$(dirname "$0")/../src"
grid="0,0.5,1.0,1.25,1.5,2.0,2.5,3.0"
np=10
outdir="../results/dev"
mkdir -p "$outdir"

# Gen <arms> <tag> [set ...] -- one generation run, tagged, with any number of --set overrides.
gen() {
    local arms="$1"
    local tag="$2"
    shift 2
    local file="dev/$tag.jsonl"
    local a=("generate.py" "--split" "dev" "--arms" "$arms" "--c-grid" "$grid" "--n-prompts" "$np" "--out" "$file" "--tag" "$tag")
    local s
    for s in "$@"; do a+=("--set" "$s"); done
    "$PY" "${a[@]}" | grep -E 'wrote|Error|Traceback' || true
}

gen "clean,naive,norm_preserving" "base"

# The loop values are written exactly as PowerShell renders them in the tags and --set arguments
# (PowerShell prints the double 1.0 as "1"), so the file names match the reported runs.
for d in mlp_mix_cond1_s0 mlp_gauss_cond1_s0; do
    for lam in 0.5 1 1.5; do
        gen "cds" "cds_${d}_lam$lam" "denoiser=$d" "lam=$lam"
    done
    for eta in 0.5 1; do
        gen "denoise_naive" "dn_${d}_eta$eta" "denoiser=$d" "eta=$eta"
    done
done
for sh in 0.01 0.05 0.2; do gen "mts" "mts_sh$sh" "shrink=$sh"; done
for k in 0.3 0.5 0.7 1; do gen "fsr" "fsr_k$k" "fsr_k=$k"; done

# one scoring pass over everything; metrics.py reads the glob directly, since concatenating with
# Set-Content would prepend a UTF-8 BOM that the JSON parser rejects
"$PY" metrics.py --gen 'dev/*.jsonl' --out scored_dev_all.csv --split dev --stages ppl,keyword,sae,dist
"$PY" pareto.py --scored scored_dev_all.csv --concept keyword_hit
