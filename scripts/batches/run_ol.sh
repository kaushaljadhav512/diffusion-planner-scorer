#!/usr/bin/env bash
# Open loop (OLS), test14-hard-100, K=16, noise 1.5: first, pdm_sl, S1, w6; seeds 0-2 (verifier seed = sampling seed).
# Simulation logs are off (~callback.simulation_log_callback): OLS needs only the metrics, and logs are ~3.5 GB per run.
cd ~/IDL_project/dps-closed-loop
eval "$(~/miniforge3/bin/conda shell.bash hook)"; conda activate diffusion_planner; source setup/config.env
export PYTHONPATH=$PWD
for s in 0 1 2; do
  for m in first pdm_sl s1:../dps-step-lane-features/runs/s1_step_lane_scalars/seed$s/best.pt w6:../dps-timestep-verifier/runs/w6_timestep/seed$s/best.pt; do
    tag=${m%%:*}
    case "$m" in
      *:*) sel=(SELECTOR=verifier VERIFIER_TAG=$tag VERIFIER_CKPT=${m#*:}) ;;
      *)   sel=(SELECTOR=$m) ;;
    esac
    echo "=== $tag seed $s ($(date "+%F %T"))"
    env K=16 NOISE=1.5 SEED=$s "${sel[@]}" bash scripts/run_sim.sh dp_k test14-hard-100 ol "~callback.simulation_log_callback" > runs/ol_${tag}_seed$s.log 2>&1
    echo "    exit $? | $(grep -hE "Number of (successful|failed) simulations" runs/ol_${tag}_seed$s.log | sed "s/.*INFO.*}  //" | tr "\n" " ")"
  done
done
python scripts/summarize_results.py --csv runs/ol_runs.csv --seeds-csv runs/ol_seeds.csv > runs/ol_summary.txt 2>&1
echo "=== all done ($(date "+%F %T"))"; touch runs/OL_DONE
