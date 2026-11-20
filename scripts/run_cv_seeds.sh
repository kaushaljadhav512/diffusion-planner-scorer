#!/usr/bin/env bash
# CV-label verifier, 3 seeds on the fixed seed-0 validation split, then the per-metric CV diagnostic per seed.
#   bash scripts/run_cv_seeds.sh                                           (w5_cv baseline: plan arch)
#   OUT=runs/w6_timestep TRAIN_ARGS="--arch timestep" bash scripts/run_cv_seeds.sh
#   env: DATA, OUT, SEEDS, EPOCHS, PY, TRAIN_ARGS (extra train.py arguments)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"; cd "$ROOT"
DATA="${DATA:-$HOME/nuplan/labels/train-labels-2k}"
OUT="${OUT:-runs/w5_cv}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-30}"
PY="${PY:-python3}"
read -r -a EXTRA <<< "${TRAIN_ARGS:-}"
mkdir -p "$OUT"
for seed in $SEEDS; do
  dir="$OUT/seed$seed"
  if [ -f "$dir/diagnostic_cv/summary.json" ]; then echo "=== $dir done, skipping"; continue; fi
  rm -rf "$dir"; mkdir -p "$dir"
  cmd=("$PY" -m diffusion_planner.scorer.train --data "$DATA" --out "$dir" --labels cv
       --epochs "$EPOCHS" --seed "$seed" --split-seed 0 "${EXTRA[@]}")
  printf '%q ' "${cmd[@]}" > "$dir/command.txt"; echo >> "$dir/command.txt"
  git rev-parse HEAD >> "$dir/command.txt"
  git status --short -- diffusion_planner scripts >> "$dir/command.txt"
  echo "=== train seed=$seed ($(date '+%F %H:%M'))"
  "${cmd[@]}" 2>&1 | tee "$dir/train.log"
  echo "=== diagnose seed=$seed ($(date '+%F %H:%M'))"
  "$PY" scripts/diagnose_verifier.py --data "$DATA" --ckpt "$dir/best.pt" --labels cv --split-seed 0 \
    --out "$dir/diagnostic_cv" 2>&1 | grep -v -i warn | tee "$dir/diagnostic_cv.log"
done
echo "=== all done ($(date '+%F %H:%M'))"
touch "$OUT/ALL_DONE"
