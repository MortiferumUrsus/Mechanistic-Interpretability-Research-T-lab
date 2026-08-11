# E4: the single TEST pass. Knobs come from configs/frozen.yaml and are not overridden here.
$ErrorActionPreference = "Stop"
$py = "$PSScriptRoot\..\.venv\Scripts\python.exe"
Push-Location "$PSScriptRoot\..\src"

$grid = "0,0.5,1.0,1.25,1.5,2.0,2.5,3.0"
$arms = "clean,naive,norm_preserving,denoise_naive,cds,mts,fsr"

& $py generate.py --split test --arms $arms --c-grid $grid --n-prompts 30 --out gen_test.jsonl
& $py metrics.py --gen gen_test.jsonl --out scored_test.csv --split test --stages ppl,keyword,sae,dist
& $py pareto.py --scored scored_test.csv --concept keyword_hit
& $py plots.py --scored scored_test.csv --concept keyword_hit

Pop-Location
