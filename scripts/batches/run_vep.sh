#!/usr/bin/env bash
# S1 with constant-velocity E (verifier_ep_cv), test14-hard-100, nr, K=16, noise 1.5, seeds 0-2 (verifier seed =
# sampling seed, as in the 2026-11-25 S1 run). Simulation logs off.
cd ~/IDL_project/dps-true-e
eval "$(~/miniforge3/bin/conda shell.bash hook)"; conda activate diffusion_planner; source setup/config.env
export PYTHONPATH=$PWD
for s in 0 1 2; do
  echo "=== verifier_ep_cv s1 seed $s ($(date "+%F %T"))"
  K=16 NOISE=1.5 SEED=$s SELECTOR=verifier_ep_cv VERIFIER_TAG=s1 VERIFIER_CKPT=../dps-step-lane-features/runs/s1_step_lane_scalars/seed$s/best.pt \
    bash scripts/run_sim.sh dp_k test14-hard-100 nr "~callback.simulation_log_callback" > runs/vep_seed$s.log 2>&1
  echo "    exit $? | $(grep -hE "Number of (successful|failed) simulations" runs/vep_seed$s.log | sed "s/.*INFO.*}  //" | tr "\n" " ")"
done
echo "=== all done ($(date "+%F %T"))"; touch runs/VEP_DONE
