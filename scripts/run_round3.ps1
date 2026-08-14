# Round three: a feature set that nothing in this work has looked at, to remove the caveat that TEST was
# opened twice. Only the two arms that matter are run -- naive steering and the direction correction -- and
# the random-rotation control, which was previously only measured on DEV.
$ErrorActionPreference = "Stop"
$py = "$PSScriptRoot\..\.venv\Scripts\python.exe"
Push-Location "$PSScriptRoot\..\src"
$grid = "0,0.5,1.0,1.25,1.5,2.0,2.5,3.0"

# select twelve fresh features under the same objective rule, appending rather than overwriting
& $py select_round3.py --seed 777 --n 12

# Retrain the correction with the holdout excluded. This step is not optional and it must come here:
# run_round2.ps1 trains dir_hot BEFORE these twelve features exist, so without retraining the round-three
# evaluation would run a correction that was trained on its own holdout. That is the exact defect the first
# attempt at this round had, and it invalidated it.
& $py train_direction.py train --rank 64 --gamma 1.0 --lr 3e-3 --steps 2000 --name dir_hot

# one pass, two arms; nothing here is tuned, every knob comes from configs/frozen.yaml
& $py generate.py --split test_r3 --arms naive,dirfix --c-grid $grid --n-prompts 30 --out gen_r3.jsonl
& $py metrics.py --gen gen_r3.jsonl --out scored_r3.csv --split test_r3 --stages ppl,keyword,sae,dist --shuffle-frac 0.15
& $py pareto.py --scored scored_r3.csv --concept keyword_hit --prefix r3_
& $py paired_at_strength.py --scored scored_r3.csv --out paired_r3.csv
& $py control_specificity.py --gen gen_r3.jsonl --split test_r3 --out control_specificity_r3.csv
& $py matched_coordinate.py --scored scored_r3.csv --prefix r3_

# the rotation control on a real test split, which round two only had on DEV
& $py generate.py --split test_r3 --arms randrot --c-grid 0,1.0,1.5,2.0,3.0 --n-prompts 30 --out gen_r3_ctrl.jsonl
& $py metrics.py --gen gen_r3_ctrl.jsonl --out scored_r3_ctrl.csv --split test_r3 --stages ppl,keyword --shuffle-frac 0.0001
& $py control_specificity.py --gen gen_r3_ctrl.jsonl --split test_r3 --out control_randrot_r3.csv
Pop-Location
