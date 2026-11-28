#!/usr/bin/env bash
# True-E ceiling: pdm_sl_ep (pdm_sl constant-velocity metrics, progress vs the human driver), test14-hard-100, nr,
# K=16, noise 1.5, seeds 0-2. Simulation logs off; scores are in aggregator_metric.
cd ~/IDL_project/dps-true-e
eval "$(~/miniforge3/bin/conda shell.bash hook)"; conda activate diffusion_planner; source setup/config.env
export PYTHONPATH=$PWD
for s in 0 1 2; do
  echo "=== pdm_sl_ep seed $s ($(date "+%F %T"))"
  K=16 NOISE=1.5 SEED=$s SELECTOR=pdm_sl_ep bash scripts/run_sim.sh dp_k test14-hard-100 nr "~callback.simulation_log_callback" > runs/true_e_seed$s.log 2>&1
  echo "    exit $? | $(grep -hE "Number of (successful|failed) simulations" runs/true_e_seed$s.log | sed "s/.*INFO.*}  //" | tr "\n" " ")"
done
echo "=== all done ($(date "+%F %T"))"; touch runs/TRUE_E_DONE
