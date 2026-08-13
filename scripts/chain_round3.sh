#!/bin/sh
# Round three, restarted after the previous chain died with its parent session mid-generation.
#
# Feature selection is not repeated: configs/features.yaml already carries test_r3 and data/splits_r3.npz
# exists, and re-running the selection would need --overwrite and would risk drawing a different set. The
# split is frozen from here on, which is the whole point of a third round.
PY=/c/Projects/Work/T-lab/01-mech-interp/.venv/Scripts/python.exe
cd /c/Projects/Work/T-lab/01-mech-interp/src
LOG=../results/round3.log
GRID="0,0.5,1.0,1.25,1.5,2.0,2.5,3.0"

echo "=== round three ===" > "$LOG"
date >> "$LOG"

echo "--- generate (two arms, fresh features) ---" >> "$LOG"
"$PY" generate.py --split test_r3 --arms naive,dirfix --c-grid $GRID --n-prompts 30 --out gen_r3.jsonl >> "$LOG" 2>&1
echo "generate exited $?" >> "$LOG"; date >> "$LOG"

echo "--- metrics ---" >> "$LOG"
"$PY" metrics.py --gen gen_r3.jsonl --out scored_r3.csv --split test_r3 --stages ppl,keyword,sae,dist --shuffle-frac 0.15 >> "$LOG" 2>&1
echo "metrics exited $?" >> "$LOG"; date >> "$LOG"

echo "--- endpoint, paired, controls ---" >> "$LOG"
"$PY" pareto.py --scored scored_r3.csv --concept keyword_hit --prefix r3_ >> "$LOG" 2>&1
"$PY" paired_at_strength.py --scored scored_r3.csv --out paired_r3.csv >> "$LOG" 2>&1
"$PY" control_specificity.py --gen gen_r3.jsonl --split test_r3 --out control_specificity_r3.csv >> "$LOG" 2>&1
"$PY" matched_coordinate.py --scored scored_r3.csv --prefix r3_ >> "$LOG" 2>&1

echo "--- rotation control on a real test split ---" >> "$LOG"
date >> "$LOG"
"$PY" generate.py --split test_r3 --arms randrot --c-grid 0,1.0,1.5,2.0,3.0 --n-prompts 30 --out gen_r3_ctrl.jsonl >> "$LOG" 2>&1
"$PY" metrics.py --gen gen_r3_ctrl.jsonl --out scored_r3_ctrl.csv --split test_r3 --stages ppl,keyword --shuffle-frac 0.0001 >> "$LOG" 2>&1
"$PY" control_specificity.py --gen gen_r3_ctrl.jsonl --split test_r3 --out control_randrot_r3.csv >> "$LOG" 2>&1

echo "=== round three complete ===" >> "$LOG"
date >> "$LOG"
