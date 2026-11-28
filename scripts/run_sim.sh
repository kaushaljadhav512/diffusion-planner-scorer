#!/usr/bin/env bash
# Run one nuPlan closed-loop simulation.
#
#   scripts/run_sim.sh <planner> <split> <challenge> [extra hydra overrides...]
#     planner:   dp          Diffusion Planner, released checkpoint, no refinement
#                dp_k        best-of-K Diffusion Planner (diffusion_planner/scorer), see K / SELECTOR / NOISE
#                pdm_closed  PDM-Closed (rule-based reference)
#     split:     val14 | test14-hard | test14-random | test14-hard-100 (scripts/make_subset_filter.py)
#                train-labels-2k (scripts/make_train_filter.py; label generation)
#     challenge: nr (non-reactive agents) | r (reactive agents) | ol (open loop: logged ego, OLS metrics)
#
# Env knobs:
#   SMOKE=N        only simulate the first N scenarios of the split (quick pipeline check)
#   THREADS=N      parallel simulations (default: min(cores, RAM_GB / 3) since each worker
#                  loads its own copy of the map; lower it if the machine runs out of memory)
#   GPU_FRAC=f     GPU share per simulation for dp (default: ngpus / threads, so every
#                  thread gets a sim; the released model is small enough for this)
#   Without an NVIDIA GPU the planner automatically runs on CPU.
#   dp_k only:
#   K=N            candidates per step (default 16)
#   SELECTOR=s     first | random | pdm | pdm_sl | pdm_sl_ep | oracle | oracle_sl | oracle_ep | label | verifier (default first)
#                    verifier: learned verifier, checkpoint VERIFIER_CKPT (default results/verifier/w3_oracle/best.pt)
#                    pdm_sl: PDM plus nuPlan's speed-limit metric, CLS weights
#                    pdm_sl_ep: pdm_sl's constant-velocity metrics with progress relative to the human driver's logged
#                               progress + ego_is_making_progress (oracle_ep without logged agent futures), nr only
#                    oracle: PDM with logged futures, nr only
#                    oracle_sl: oracle with pdm_sl's score (the verifier's current training target), nr only
#                    oracle_ep: oracle_sl with nuPlan's expert-relative progress + ego_is_making_progress, nr only;
#                               per-step picks under both definitions go to ORACLE_STATS_DIR if set
#                    label:  drives with PDM and records verifier labels to LABEL_DIR, nr only (diffusion_planner/scorer/labels.py)
#   VERIFIER_TAG=t verifier only: appended to the run name (dp_k<K>_verifier_<t>...) so runs of different checkpoints
#                  are kept apart, e.g. VERIFIER_TAG=s1
#   LABEL_DIR=d    label output dir (default $NUPLAN_HOME/labels/<split>); LABEL_DRIVE=pdm | pdm_sl (default pdm_sl)
#   NOISE=f        initial diffusion noise std (default 0.5, upstream's value)
#   NOISE=a,b      mixed: candidate 0 at std a (a=0.5 is vanilla DP's own plan), candidates 1..K-1 at std b
#   SEED=n         torch seed, reset per scenario (default 0)
#   Runs are stored as dp_k<K>_<selector>[_noise<NOISE>][_seed<SEED>] (mixed noise: _noise<a>-<b>) so different
#   settings don't overwrite each other in summarize_results.py.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../setup/config.env"

PLANNER="${1:?planner: dp | dp_k | pdm_closed}"
SPLIT="${2:?split: val14 | test14-hard | test14-random}"
CH="${3:?challenge: nr | r | ol}"
shift 3

case "$CH" in
  nr) CHALLENGE=closed_loop_nonreactive_agents ;;
  r)  CHALLENGE=closed_loop_reactive_agents ;;
  ol) CHALLENGE=open_loop_boxes ;;
  *)  echo "challenge must be nr, r or ol"; exit 1 ;;
esac
# Val14 and train-* (scripts/make_train_filter.py) scenarios live in trainval/; Test14-* come from the test DBs.
case "$SPLIT" in
  val14|train-*) SCENARIO_BUILDER=nuplan ;;
  *)             SCENARIO_BUILDER=nuplan_challenge ;;
esac

RAM_GB="$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)"
THREADS="${THREADS:-$(( RAM_GB / 3 < $(nproc) ? RAM_GB / 3 : $(nproc) ))}"
[ "$THREADS" -ge 1 ] || THREADS=1
NGPU="$( (nvidia-smi -L 2>/dev/null || true) | grep -c '^GPU' || true)"
GPU_FRAC="${GPU_FRAC:-$(python3 -c "print(round(max($NGPU,1)/$THREADS, 4))")}"
if [ "$NGPU" -eq 0 ]; then
  GPU_ARGS=(gpu=false number_of_gpus_allocated_per_simulation=0)
  DP_DEVICE=cpu
else
  GPU_ARGS=(number_of_gpus_allocated_per_simulation="$GPU_FRAC")
  DP_DEVICE=cuda
fi
NAME="$PLANNER"

# Use Diffusion Planner's scenario_filter configs for every planner so all rows of the
# results table are evaluated on exactly the same scenario tokens.
SEARCH="pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config"
case "$PLANNER" in
  dp)
    PLANNER_ARGS=(
      planner=diffusion_planner
      planner.diffusion_planner.config.args_file="$DP_ROOT/checkpoints/args.json"
      planner.diffusion_planner.ckpt_path="$DP_ROOT/checkpoints/model.pth"
      planner.diffusion_planner.device="$DP_DEVICE"
      "${GPU_ARGS[@]}"
    )
    ;;
  dp_k)
    K="${K:-16}"
    SELECTOR="${SELECTOR:-first}"
    NOISE="${NOISE:-0.5}"
    SEED="${SEED:-0}"
    SELECTOR_ARGS=()
    case "$SELECTOR" in
      first)  SELECTOR_TARGET=diffusion_planner.scorer.selectors.FirstSample ;;
      random) SELECTOR_TARGET=diffusion_planner.scorer.selectors.Random ;;
      pdm)    SELECTOR_TARGET=diffusion_planner.scorer.selectors.PDMSelector ;;
      pdm_sl) SELECTOR_TARGET=diffusion_planner.scorer.selectors.PDMSpeedLimitSelector ;;
      pdm_sl_ep)
        [ "$CH" = nr ] || { echo "SELECTOR=pdm_sl_ep needs nr: the human driver's progress is read from the log"; exit 1; }
        SELECTOR_TARGET=diffusion_planner.scorer.selectors.PDMExpertProgressSelector ;;
      oracle)
        [ "$CH" = nr ] || { echo "SELECTOR=oracle needs nr: reactive agents leave the logged future"; exit 1; }
        SELECTOR_TARGET=diffusion_planner.scorer.selectors.OracleSelector ;;
      oracle_sl)
        [ "$CH" = nr ] || { echo "SELECTOR=oracle_sl needs nr: reactive agents leave the logged future"; exit 1; }
        SELECTOR_TARGET=diffusion_planner.scorer.selectors.OracleSpeedLimitSelector ;;
      oracle_ep)
        [ "$CH" = nr ] || { echo "SELECTOR=oracle_ep needs nr: reactive agents leave the logged future"; exit 1; }
        SELECTOR_TARGET=diffusion_planner.scorer.selectors.OracleExpertProgressSelector
        [ -z "${ORACLE_STATS_DIR:-}" ] || SELECTOR_ARGS+=("++planner.diffusion_planner_k.selector.stats_dir=$ORACLE_STATS_DIR") ;;
      label)
        [ "$CH" = nr ] || { echo "SELECTOR=label needs nr: the oracle labels use the logged future"; exit 1; }
        SELECTOR_TARGET=diffusion_planner.scorer.labels.LabelRecorder
        SELECTOR_ARGS+=(
          "++planner.diffusion_planner_k.selector.out_dir=${LABEL_DIR:-$NUPLAN_HOME/labels/$SPLIT}"
          "++planner.diffusion_planner_k.selector.drive=${LABEL_DRIVE:-pdm_sl}"
        ) ;;
      verifier)
        SELECTOR_TARGET=diffusion_planner.scorer.selectors.VerifierSelector
        SELECTOR_ARGS+=(
          "++planner.diffusion_planner_k.selector.ckpt_path=$DP_ROOT/${VERIFIER_CKPT:-results/verifier/w3_oracle/best.pt}"
          "++planner.diffusion_planner_k.selector.args_file=$DP_ROOT/checkpoints/args.json"
          "++planner.diffusion_planner_k.selector.device=$DP_DEVICE"
        ) ;;
      *) echo "unknown SELECTOR $SELECTOR (first | random | pdm | pdm_sl | pdm_sl_ep | oracle | oracle_sl | oracle_ep | label | verifier)"; exit 1 ;;
    esac
    case "$NOISE" in
      *,*) NOISE_FIRST="${NOISE%%,*}"; NOISE_REST="${NOISE#*,}" ;;
      *)   NOISE_FIRST=null; NOISE_REST="$NOISE" ;;
    esac
    NAME="dp_k${K}_${SELECTOR}"
    [ "$SELECTOR" != verifier ] || [ -z "${VERIFIER_TAG:-}" ] || NAME="${NAME}_${VERIFIER_TAG}"
    [ "$NOISE" = "0.5" ] || NAME="${NAME}_noise${NOISE/,/-}"
    [ "$SEED" = "0" ] || NAME="${NAME}_seed${SEED}"
    PLANNER_ARGS=(
      planner=diffusion_planner_k
      planner.diffusion_planner_k.config.args_file="$DP_ROOT/checkpoints/args.json"
      planner.diffusion_planner_k.ckpt_path="$DP_ROOT/checkpoints/model.pth"
      planner.diffusion_planner_k.device="$DP_DEVICE"
      planner.diffusion_planner_k.K="$K"
      planner.diffusion_planner_k.noise_scale="$NOISE_REST"
      planner.diffusion_planner_k.noise_scale_first="$NOISE_FIRST"
      planner.diffusion_planner_k.seed="$SEED"
      planner.diffusion_planner_k.selector._target_="$SELECTOR_TARGET"
      "${SELECTOR_ARGS[@]}"
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
UID_="$NAME/$SPLIT$TAG/$(date +%Y-%m-%d-%H-%M-%S)"
echo "[run_sim] $NAME | $SPLIT$TAG | $CHALLENGE | threads=$THREADS (ram=${RAM_GB}GB) gpus=$NGPU gpu_frac=$GPU_FRAC"

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
