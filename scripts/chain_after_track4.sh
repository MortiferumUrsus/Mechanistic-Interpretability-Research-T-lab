#!/bin/sh
# Start round three as soon as track 4's night chain releases the GPU, so the card is not idle overnight.
#
# The wait condition is the file the chain writes last, not a timer: slice 2b's CSV appears only after its
# final cell, and the analysis step that follows is CPU-only, so there is no contention once it exists.
# Round three needs the GPU for generation and for the metric stages, which is why it waits at all.
#
# Round three is the answer to the one caveat the report cannot argue away: TEST was opened twice, because
# round two chose its method after seeing round one's TEST numbers. Fresh features fix that; better wording
# does not. The features were already selected (configs/features.yaml test_r3, data/splits_r3.npz).

TRACK4=/c/Projects/Work/T-lab/04-lifelong-agents/results
LOG=/c/Projects/Work/T-lab/01-mech-interp/results/round3_chain.log

echo "=== waiting for track 4's chain to finish (slice_corrupt_a000.csv) ===" > "$LOG"
date >> "$LOG"

# Bounded wait: if the chain dies, this must not sit here for ever holding the queue.
waited=0
while [ ! -f "$TRACK4/slice_corrupt_a000.csv" ]; do
    sleep 120
    waited=$((waited + 120))
    if [ "$waited" -gt 28800 ]; then
        echo "gave up after 8 hours: track 4's chain has not produced slice_corrupt_a000.csv" >> "$LOG"
        echo "not starting round three; check the chain before running scripts/run_round3.ps1 by hand" >> "$LOG"
        exit 1
    fi
done

echo "track 4 finished after ${waited}s of waiting; letting the GPU settle" >> "$LOG"
sleep 60

echo "=== round three ===" >> "$LOG"
date >> "$LOG"
powershell.exe -NoProfile -ExecutionPolicy Bypass \
    -File 'C:\Projects\Work\T-lab\01-mech-interp\scripts\run_round3.ps1' >> "$LOG" 2>&1
status=$?
echo "run_round3.ps1 exited with $status" >> "$LOG"
date >> "$LOG"
echo "=== round three chain complete ===" >> "$LOG"
