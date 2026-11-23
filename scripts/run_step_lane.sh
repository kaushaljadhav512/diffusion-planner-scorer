#!/usr/bin/env bash
# Per-step local lane features on the timestep verifier (w6 setup: cv labels, split seed 0, 30 epochs; the decoder,
# heads and DP's 70 lane tokens unchanged). One change per arm, per seed:
#   S1  --step-lane scalars   exact geometry per step (signed corner distances, route, lane under ego)
#                                                         -> runs/s1_step_lane_scalars/seed<s>
#   S2  --step-lane segments  S1 plus an attention pool over the nearest lane segments
#                                                         -> runs/s2_step_lane_segments/seed<s>
# Compared with w6 (0.0352 +- 0.0013, drivable pair acc 0.57) on regret, top-1 and drivable within-scene pair
# accuracy; each finished run gets the per-metric CV diagnostic. Seeds run in order (seed 0 of both arms first).
# Re-running skips finished runs.
#   bash scripts/run_step_lane.sh
#   env: DATA, SEEDS, EPOCHS, PY
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"; cd "$ROOT"
DATA="${DATA:-$HOME/nuplan/labels/train-labels-2k}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-30}"
PY="${PY:-python3}"

gpu_busy() { # any GPU compute process (aegis jobs all run as the shared account, so owners can't be told apart)
  [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]
}

train() { # train <dir> <train.py args...>
  local dir="$1"; shift
  if [ -f "$dir/best_metrics.json" ] && [ -f "$dir/diagnostic_cv/summary.json" ]; then
    echo "=== $dir done, skipping"; return
  fi
  while gpu_busy; do echo "=== GPU busy with another job ($(date '+%F %H:%M'))"; sleep 300; done
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

mkdir -p runs
for seed in $SEEDS; do
  train "runs/s1_step_lane_scalars/seed$seed" --epochs "$EPOCHS" --seed "$seed" --step-lane scalars
  train "runs/s2_step_lane_segments/seed$seed" --epochs "$EPOCHS" --seed "$seed" --step-lane segments
done
echo "=== all done ($(date '+%F %H:%M'))"
touch runs/STEP_LANE_DONE
