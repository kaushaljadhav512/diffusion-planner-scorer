#!/usr/bin/env bash
# Headroom run (task 1.10): best-of-K Diffusion Planner with every selector on the fixed subset, then the table
# (CLS, gain over FirstSample, per-step sampling / selection time) printed and written to results/.
#   scripts/run_headroom.sh                                # K=16: nr first/random/pdm/oracle, r first/random/pdm
#   K=8 NOISE=1.0 scripts/run_headroom.sh                  # another K / noise (1.11 rerun)
#   SMOKE=5 SELECTORS="first oracle" CHALLENGES=nr scripts/run_headroom.sh
# Several hours at full size; run inside tmux. The subset filter comes from scripts/make_subset_filter.py.
set -uo pipefail  # no -e: one failed run shouldn't kill the rest of the grid
HERE="$(dirname "${BASH_SOURCE[0]}")"
source "$HERE/../setup/config.env"

SPLIT="${SPLIT:-test14-hard-100}"
SELECTORS="${SELECTORS:-first random pdm oracle}"  # first first: the others are compared against it
CHALLENGES="${CHALLENGES:-nr r}"
export K="${K:-16}" NOISE="${NOISE:-0.5}"

[ -f "$DP_ROOT/diffusion_planner/config/scenario_filter/$SPLIT.yaml" ] \
  || { echo "no scenario filter $SPLIT; run scripts/make_subset_filter.py first"; exit 1; }

mkdir -p "$NUPLAN_EXP_ROOT/logs"
failed=()
for c in $CHALLENGES; do for s in $SELECTORS; do
  [ "$s" = oracle ] && [ "$c" = r ] && continue  # logged futures are only exact without reactive agents
  log="$NUPLAN_EXP_ROOT/logs/dp_k${K}_${s}_noise${NOISE}_${SPLIT}_${c}${SMOKE:+_smoke$SMOKE}_$(date +%m%d-%H%M).log"
  echo "=== K=$K $s / $SPLIT / $c  (log: $log)"
  start=$SECONDS
  if SELECTOR="$s" bash "$HERE/run_sim.sh" dp_k "$SPLIT" "$c" >"$log" 2>&1; then
    echo "    ok in $(( (SECONDS-start)/60 )) min"
  else
    echo "    FAILED after $(( (SECONDS-start)/60 )) min - tail of log:"; tail -n 15 "$log" | sed 's/^/      /'
    failed+=("$s/$c")
  fi
done; done

# Table of every run on this split (all K / noise settings so far); split dir as in run_sim.sh
SPLIT_DIR="$SPLIT${SMOKE:+_smoke$SMOKE}"
mkdir -p "$PROJ_ROOT/results"
python "$HERE/summarize_results.py" --split "$SPLIT_DIR" --csv "$PROJ_ROOT/results/headroom_${SPLIT_DIR}.csv"
[ ${#failed[@]} -eq 0 ] || { echo "Failed runs: ${failed[*]}"; exit 1; }
