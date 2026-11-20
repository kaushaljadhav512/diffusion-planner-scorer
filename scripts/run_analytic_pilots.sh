#!/usr/bin/env bash
# Dense analytic drivable-area labels on the unchanged timestep verifier (w6 setup, cv labels, split seed 0), per seed:
#   A1  multi-task: w6 loss + --analytic-weight from the start, EPOCHS epochs        -> runs/a1_analytic_multitask/seed<s>
#   A2  stage A: --analytic-only for PRE_EPOCHS epochs                                -> runs/a2_stage_a/seed<s>
#       stage B: w6 loss for EPOCHS epochs from stage A's best checkpoint             -> runs/a2_analytic_pretrain/seed<s>
# Seeds run in order (seed 0 = the pilots first), each finished run gets the per-metric CV diagnostic.
#   bash scripts/run_analytic_pilots.sh
#   env: DATA, SEEDS, EPOCHS, PRE_EPOCHS, ANALYTIC_WEIGHT, PY
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"; cd "$ROOT"
DATA="${DATA:-$HOME/nuplan/labels/train-labels-2k}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-30}"
PRE_EPOCHS="${PRE_EPOCHS:-10}"
ANALYTIC_WEIGHT="${ANALYTIC_WEIGHT:-0.3}"
PY="${PY:-python3}"

train() { # train <dir> <diagnose: 1|0> <train.py args...>
  local dir="$1" diagnose="$2"; shift 2
  if [ -f "$dir/best_metrics.json" ] && { [ "$diagnose" = 0 ] || [ -f "$dir/diagnostic_cv/summary.json" ]; }; then
    echo "=== $dir done, skipping"; return
  fi
  rm -rf "$dir"; mkdir -p "$dir"
  local cmd=("$PY" -m diffusion_planner.scorer.train --data "$DATA" --out "$dir" --labels cv --split-seed 0
             --arch timestep "$@")
  printf '%q ' "${cmd[@]}" > "$dir/command.txt"; echo >> "$dir/command.txt"
  git rev-parse HEAD >> "$dir/command.txt"
  git status --short -- diffusion_planner scripts >> "$dir/command.txt"
  echo "=== train $dir ($(date '+%F %H:%M'))"
  "${cmd[@]}" 2>&1 | tee "$dir/train.log"
  if [ "$diagnose" = 1 ]; then
    echo "=== diagnose $dir ($(date '+%F %H:%M'))"
    "$PY" scripts/diagnose_verifier.py --data "$DATA" --ckpt "$dir/best.pt" --labels cv --split-seed 0 \
      --out "$dir/diagnostic_cv" 2>&1 | grep -v -i warn | tee "$dir/diagnostic_cv.log"
  fi
}

for seed in $SEEDS; do
  train "runs/a1_analytic_multitask/seed$seed" 1 --epochs "$EPOCHS" --seed "$seed" --analytic-weight "$ANALYTIC_WEIGHT"
  train "runs/a2_stage_a/seed$seed" 0 --epochs "$PRE_EPOCHS" --seed "$seed" --analytic-only
  train "runs/a2_analytic_pretrain/seed$seed" 1 --epochs "$EPOCHS" --seed "$seed" \
    --init-from "runs/a2_stage_a/seed$seed/best.pt"
done
echo "=== all done ($(date '+%F %H:%M'))"
touch runs/ANALYTIC_PILOTS_DONE
