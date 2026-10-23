#!/usr/bin/env bash
# Run the full baseline grid, then print the results table.
#   scripts/run_baselines.sh                     # everything (several hours; run inside tmux)
#   PLANNERS="dp" SPLITS="val14" scripts/run_baselines.sh
#   SMOKE=20 scripts/run_baselines.sh            # quick end-to-end check on 20 scenarios each
set -uo pipefail  # no -e: one failed run shouldn't kill the rest of the grid
HERE="$(dirname "${BASH_SOURCE[0]}")"
source "$HERE/../setup/config.env"

PLANNERS="${PLANNERS:-dp pdm_closed}"
SPLITS="${SPLITS:-val14 test14-hard}"
CHALLENGES="${CHALLENGES:-nr r}"

mkdir -p "$NUPLAN_EXP_ROOT/logs"
failed=()
for p in $PLANNERS; do for s in $SPLITS; do for c in $CHALLENGES; do
  log="$NUPLAN_EXP_ROOT/logs/${p}_${s}_${c}${SMOKE:+_smoke$SMOKE}_$(date +%m%d-%H%M).log"
  echo "=== $p / $s / $c  (log: $log)"
  start=$SECONDS
  if bash "$HERE/run_sim.sh" "$p" "$s" "$c" >"$log" 2>&1; then
    echo "    ok in $(( (SECONDS-start)/60 )) min"
  else
    echo "    FAILED after $(( (SECONDS-start)/60 )) min - tail of log:"; tail -n 15 "$log" | sed 's/^/      /'
    failed+=("$p/$s/$c")
  fi
done; done; done

python "$HERE/summarize_results.py"
[ ${#failed[@]} -eq 0 ] || { echo "Failed runs: ${failed[*]}"; exit 1; }
