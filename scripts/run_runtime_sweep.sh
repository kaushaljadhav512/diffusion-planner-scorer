#!/usr/bin/env bash
# PDM selection cost vs K (task 2.13): one simulation worker (no GPU contention) on a few scenarios of the subset,
# per K: FirstSample (sampling only) and PDMSelector. select_ms / sample_ms come from the planner's runtime report.
#   scripts/run_runtime_sweep.sh                     # K = 4 8 16 32 64, 10 scenarios, noise 1.5
#   KS="16 64" SCENARIOS=5 scripts/run_runtime_sweep.sh
# Sampling alone is timed more cleanly by scripts/bench_runtime.py; this script is for the selection column.
set -uo pipefail
HERE="$(dirname "${BASH_SOURCE[0]}")"
source "$HERE/../setup/config.env"

KS="${KS:-4 8 16 32 64}"
SCENARIOS="${SCENARIOS:-10}"
SELECTORS="${SELECTORS:-first pdm}"
export NOISE="${NOISE:-1.5}" SEED="${SEED:-0}"
SPLIT=test14-hard-100

mkdir -p "$NUPLAN_EXP_ROOT/logs"
failed=()
for k in $KS; do for s in $SELECTORS; do
  log="$NUPLAN_EXP_ROOT/logs/runtime_k${k}_${s}_$(date +%m%d-%H%M).log"
  echo "=== K=$k $s (log: $log)"
  if K="$k" SELECTOR="$s" THREADS=1 SMOKE="$SCENARIOS" bash "$HERE/run_sim.sh" dp_k "$SPLIT" nr >"$log" 2>&1; then
    echo "    ok"
  else
    echo "    FAILED - tail of log:"; tail -n 15 "$log" | sed 's/^/      /'; failed+=("K$k/$s")
  fi
done; done

python "$HERE/summarize_results.py" --split "${SPLIT}_smoke${SCENARIOS}" --csv "$PROJ_ROOT/results/runtime_k_sweep.csv"
[ ${#failed[@]} -eq 0 ] || { echo "Failed runs: ${failed[*]}"; exit 1; }
