#!/usr/bin/env bash
# Free E baselines after the true-E ceiling: pdm_sl_ep_<median|ca|cv> at noise 1.5 and pdm_sl_ep_first at noise
# 0.5,1.5 (candidate 0 = vanilla DP plan); test14-hard-100, nr, K=16, seeds 0-2, simulation logs off.
cd ~/IDL_project/dps-true-e
while [ ! -e runs/TRUE_E_DONE ]; do sleep 30; done
eval "$(~/miniforge3/bin/conda shell.bash hook)"; conda activate diffusion_planner; source setup/config.env
export PYTHONPATH=$PWD
git fetch -q runs/true-e2.bundle true-e-ceiling && git merge -q --ff-only FETCH_HEAD || { echo "update failed"; exit 1; }
echo "=== at $(git log --oneline -1)"
CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_selectors.py tests/test_metrics.py 2>&1 | tail -1
for sel in median ca cv first; do
  noise=1.5; [ $sel = first ] && noise=0.5,1.5
  for s in 0 1 2; do
    echo "=== pdm_sl_ep_$sel noise $noise seed $s ($(date "+%F %T"))"
    K=16 NOISE=$noise SEED=$s SELECTOR=pdm_sl_ep_$sel bash scripts/run_sim.sh dp_k test14-hard-100 nr "~callback.simulation_log_callback" > runs/free_${sel}_seed$s.log 2>&1
    echo "    exit $? | $(grep -hE "Number of (successful|failed) simulations" runs/free_${sel}_seed$s.log | sed "s/.*INFO.*}  //" | tr "\n" " ")"
  done
done
echo "=== all done ($(date "+%F %T"))"; touch runs/FREE_DONE
