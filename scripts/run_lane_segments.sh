#!/usr/bin/env bash
# Lane segment tokens on the timestep verifier (w6 setup + --lane-segments, cv labels, split seed 0), per seed:
#   L1  w6 loss, each DP lane input cut into SEGMENTS trainable tokens added to the memory -> runs/l1_lane_segments/seed<s>
# Waits first for WAIT_FOR (the analytic pilots' done marker), then until no process is using the GPU.
# Each finished run gets the per-metric CV diagnostic. Re-running skips finished runs.
#   bash scripts/run_lane_segments.sh
#   env: DATA, SEEDS, EPOCHS, SEGMENTS, WAIT_FOR, PY
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"; cd "$ROOT"
DATA="${DATA:-$HOME/nuplan/labels/train-labels-2k}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-30}"
SEGMENTS="${SEGMENTS:-4}"
WAIT_FOR="${WAIT_FOR:-$HOME/IDL_project/dps-analytic-pretrain/runs/ANALYTIC_PILOTS_DONE}"
PY="${PY:-python3}"

gpu_busy() { # any GPU compute process (aegis jobs all run as the shared account, so owners can't be told apart)
  [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]
}
echo "=== waiting for $WAIT_FOR ($(date '+%F %H:%M'))"
while [ ! -f "$WAIT_FOR" ]; do sleep 120; done
while gpu_busy; do echo "=== GPU busy with another job ($(date '+%F %H:%M'))"; sleep 300; done

train() { # train <dir> <train.py args...>
  local dir="$1"; shift
  if [ -f "$dir/best_metrics.json" ] && [ -f "$dir/diagnostic_cv/summary.json" ]; then
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
  echo "=== diagnose $dir ($(date '+%F %H:%M'))"
  "$PY" scripts/diagnose_verifier.py --data "$DATA" --ckpt "$dir/best.pt" --labels cv --split-seed 0 \
    --out "$dir/diagnostic_cv" 2>&1 | grep -v -i warn | tee "$dir/diagnostic_cv.log"
}

for seed in $SEEDS; do
  train "runs/l1_lane_segments/seed$seed" --epochs "$EPOCHS" --seed "$seed" --lane-segments "$SEGMENTS"
done
echo "=== all done ($(date '+%F %H:%M'))"
touch runs/LANE_SEGMENTS_DONE
