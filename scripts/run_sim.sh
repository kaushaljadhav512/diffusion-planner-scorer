#!/usr/bin/env bash
# Run one nuPlan closed-loop simulation.
#
#   scripts/run_sim.sh <planner> <split> <challenge> [extra hydra overrides...]
#     planner:   dp          Diffusion Planner, released checkpoint, no refinement
#                pdm_closed  PDM-Closed (rule-based reference)
#     split:     val14 | test14-hard | test14-random
#     challenge: nr (non-reactive agents) | r (reactive agents)
#
# Env knobs:
#   SMOKE=N        only simulate the first N scenarios of the split (quick pipeline check)
#   THREADS=N      parallel simulations (default: min(cores, RAM_GB / 3) since each worker
#                  loads its own copy of the map; lower it if the machine runs out of memory)
#   GPU_FRAC=f     GPU share per simulation for dp (default: ngpus / threads, so every
#                  thread gets a sim; the released model is small enough for this)
#   Without an NVIDIA GPU the planner automatically runs on CPU.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../setup/config.env"

PLANNER="${1:?planner: dp | pdm_closed}"
SPLIT="${2:?split: val14 | test14-hard | test14-random}"
CH="${3:?challenge: nr | r}"
shift 3

case "$CH" in
  nr) CHALLENGE=closed_loop_nonreactive_agents ;;
  r)  CHALLENGE=closed_loop_reactive_agents ;;
  *)  echo "challenge must be nr or r"; exit 1 ;;
esac
# Val14 scenarios live in the val DBs (trainval/); Test14-* come from the test DBs.
[ "$SPLIT" = "val14" ] && SCENARIO_BUILDER=nuplan || SCENARIO_BUILDER=nuplan_challenge

RAM_GB="$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)"
THREADS="${THREADS:-$(( RAM_GB / 3 < $(nproc) ? RAM_GB / 3 : $(nproc) ))}"
[ "$THREADS" -ge 1 ] || THREADS=1
NGPU="$( (nvidia-smi -L 2>/dev/null || true) | grep -c '^GPU' || true)"
GPU_FRAC="${GPU_FRAC:-$(python3 -c "print(round(max($NGPU,1)/$THREADS, 4))")}"
if [ "$NGPU" -eq 0 ]; then
  GPU_ARGS=(gpu=false number_of_gpus_allocated_per_simulation=0 planner.diffusion_planner.device=cpu)
else
  GPU_ARGS=(number_of_gpus_allocated_per_simulation="$GPU_FRAC")
fi

# Use Diffusion Planner's scenario_filter configs for every planner so all rows of the
# results table are evaluated on exactly the same scenario tokens.
SEARCH="pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config"
case "$PLANNER" in
  dp)
    PLANNER_ARGS=(
      planner=diffusion_planner
      planner.diffusion_planner.config.args_file="$DP_ROOT/checkpoints/args.json"
      planner.diffusion_planner.ckpt_path="$DP_ROOT/checkpoints/model.pth"
      "${GPU_ARGS[@]}"
    )
    ;;
  pdm_closed)
    SEARCH="$SEARCH, pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation"
    PLANNER_ARGS=(planner=pdm_closed_planner number_of_gpus_allocated_per_simulation=0)
    ;;
  *) echo "unknown planner $PLANNER"; exit 1 ;;
esac
SEARCH="[$SEARCH, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"

EXTRA=()
TAG=""
if [ -n "${SMOKE:-}" ]; then
  EXTRA+=(scenario_filter.limit_total_scenarios="$SMOKE")
  TAG="_smoke$SMOKE"
fi

# Output: $NUPLAN_EXP_ROOT/exp/simulation/<challenge>/<planner>/<split><tag>/<timestamp>/
UID_="$PLANNER/$SPLIT$TAG/$(date +%Y-%m-%d-%H-%M-%S)"
echo "[run_sim] $PLANNER | $SPLIT$TAG | $CHALLENGE | threads=$THREADS (ram=${RAM_GB}GB) gpus=$NGPU gpu_frac=$GPU_FRAC"

export HYDRA_FULL_ERROR=1
python "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
  +simulation="$CHALLENGE" \
  "${PLANNER_ARGS[@]}" \
  scenario_builder="$SCENARIO_BUILDER" \
  scenario_filter="$SPLIT" \
  experiment_uid="$UID_" \
  verbose=true \
  worker=ray_distributed \
  worker.threads_per_node="$THREADS" \
  distributed_mode=SINGLE_NODE \
  enable_simulation_progress_bar=true \
  hydra.searchpath="$SEARCH" \
  "${EXTRA[@]}" "$@"
