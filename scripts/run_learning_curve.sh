#!/usr/bin/env bash
# Verifier learning curve: train on a fraction of the training scenarios, evaluate on the same validation
# scenarios (split seed 0, as w3_oracle), and summarize held-out regret / top-1 against the fraction.
#
#   DATA=~/nuplan/labels/train-labels-2k bash scripts/run_learning_curve.sh [extra train.py args]
#
# Every run gets the same number of gradient steps as 20 epochs on all the data (epochs = 20 / frac, at most 200),
# so small fractions are not undertrained; the best epoch on validation is kept, as in train.py.
# Env: DATA (label dir), OUT (default runs/learning_curve), FRACS, SEEDS (seed = subset and init).
# Finished runs (best_metrics.json present) are skipped, so the script can be re-run after an interruption.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"

DATA="${DATA:-$NUPLAN_HOME/labels/train-labels-2k}"
OUT="${OUT:-runs/learning_curve}"
FRACS="${FRACS:-0.1 0.25 0.5 1.0}"
SEEDS="${SEEDS:-0 1 2}"
PY="${PY:-python3}"

mkdir -p "$OUT"
for seed in $SEEDS; do for frac in $FRACS; do
  dir="$OUT/frac${frac}_seed${seed}"
  if [ -f "$dir/best_metrics.json" ]; then echo "=== $dir done, skipping"; continue; fi
  epochs=$("$PY" -c "print(min(200, round(20 / $frac)))")
  rm -rf "$dir"  # log.jsonl is opened in append mode: start an interrupted run from scratch
  mkdir -p "$dir"
  echo "=== frac=$frac seed=$seed epochs=$epochs  ($(date '+%H:%M'))"
  "$PY" -m diffusion_planner.scorer.train --data "$DATA" --out "$dir" \
    --train-frac "$frac" --seed "$seed" --split-seed 0 --epochs "$epochs" "$@" 2>&1 | tee "$dir/train.log"
done; done

"$PY" "$HERE/summarize_learning_curve.py" "$OUT" --csv "$ROOT/results/verifier/learning_curve.csv"
