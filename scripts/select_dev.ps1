# DEV knob selection. Every candidate is judged by the same metric the report uses, on DEV
# features only. TEST is never generated here.
$ErrorActionPreference = "Continue"
$py = "$PSScriptRoot\..\.venv\Scripts\python.exe"
Push-Location "$PSScriptRoot\..\src"
$grid = "0,0.5,1.0,1.25,1.5,2.0,2.5,3.0"
$np = 10
$outdir = "..\results\dev"
New-Item -ItemType Directory -Force $outdir | Out-Null

function Gen($arms, $tag, $sets) {
    $file = "dev/$tag.jsonl"
    $a = @("generate.py", "--split", "dev", "--arms", $arms, "--c-grid", $grid, "--n-prompts", "$np", "--out", $file, "--tag", $tag)
    foreach ($s in $sets) { $a += @("--set", $s) }
    & $py @a | Select-String -Pattern 'wrote|Error|Traceback'
}

Gen "clean,naive,norm_preserving" "base" @()

foreach ($d in @("mlp_mix_cond1_s0", "mlp_gauss_cond1_s0")) {
    foreach ($lam in @(0.5, 1.0, 1.5)) {
        Gen "cds" "cds_${d}_lam$lam" @("denoiser=$d", "lam=$lam")
    }
    foreach ($eta in @(0.5, 1.0)) {
        Gen "denoise_naive" "dn_${d}_eta$eta" @("denoiser=$d", "eta=$eta")
    }
}
foreach ($sh in @(0.01, 0.05, 0.2)) { Gen "mts" "mts_sh$sh" @("shrink=$sh") }
foreach ($k in @(0.3, 0.5, 0.7, 1.0)) { Gen "fsr" "fsr_k$k" @("fsr_k=$k") }

# one scoring pass over everything
Get-Content (Get-ChildItem "$outdir\*.jsonl" | ForEach-Object { $_.FullName }) | Set-Content "..\results\gen_dev_all.jsonl" -Encoding utf8
& $py metrics.py --gen gen_dev_all.jsonl --out scored_dev_all.csv --split dev --stages ppl,keyword,sae,dist
& $py pareto.py --scored scored_dev_all.csv --concept keyword_hit

Pop-Location
