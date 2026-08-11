# Round two: learn a direction correction, select it on DEV, evaluate once on TEST.
# Declared as a second round because the method was chosen after seeing round one.
$ErrorActionPreference = "Stop"
$py = "$PSScriptRoot\..\.venv\Scripts\python.exe"
Push-Location "$PSScriptRoot\..\src"
$grid = "0,0.5,1.0,1.25,1.5,2.0,2.5,3.0"

# train the shared low-rank correction on FIT directions only
& $py train_direction.py train --rank 64 --gamma 1.0 --lr 3e-3 --steps 2000 --name dir_hot

# select on held-out DEV directions, in response space (no sampling, no judge)
& $py train_direction.py eval --checkpoints dir_hot --split dev

# text-level check on DEV, then the specificity control
& $py generate.py --split dev --arms naive,dirfix --c-grid $grid --n-prompts 10 --out gen_dev_r2.jsonl
& $py metrics.py --gen gen_dev_r2.jsonl --out scored_dev_r2.csv --split dev --stages ppl,keyword,sae --shuffle-frac 0.0001
& $py control_specificity.py --gen gen_dev_r2.jsonl --split dev --out control_specificity_dev.csv

# one pass on TEST
& $py generate.py --split test --arms naive,dirfix --c-grid $grid --n-prompts 30 --out gen_test_r2.jsonl
& $py metrics.py --gen gen_test_r2.jsonl --out scored_test_r2.csv --split test --stages ppl,keyword,sae,dist --shuffle-frac 0.15
& $py pareto.py --scored scored_test_r2.csv --concept keyword_hit
& $py control_specificity.py --gen gen_test_r2.jsonl --split test --out control_specificity_test.csv

Pop-Location
