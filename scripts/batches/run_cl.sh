#!/usr/bin/env bash
# Closed loop, test14-hard-100, non-reactive, K=16, noise 1.5: S1 vs w6, seeds 0-2 (verifier seed = sampling seed).
# Baselines already on disk: dp_k16_first_noise1.5[_seed1|_seed2], dp_k16_pdm_sl_noise1.5[_seed1|_seed2].
cd ~/IDL_project/dps-closed-loop
eval "$(~/miniforge3/bin/conda shell.bash hook)"; conda activate diffusion_planner; source setup/config.env
export PYTHONPATH=$PWD
for s in 0 1 2; do
  for m in s1:../dps-step-lane-features/runs/s1_step_lane_scalars/seed$s/best.pt w6:../dps-timestep-verifier/runs/w6_timestep/seed$s/best.pt; do
    echo "=== ${m%%:*} seed $s ($(date "+%F %T"))"
    K=16 NOISE=1.5 SEED=$s SELECTOR=verifier VERIFIER_TAG=${m%%:*} VERIFIER_CKPT=${m#*:} \
      bash scripts/run_sim.sh dp_k test14-hard-100 nr > runs/cl_${m%%:*}_seed$s.log 2>&1
    echo "    exit $? | $(grep -hE "Number of (successful|failed) simulations" runs/cl_${m%%:*}_seed$s.log | sed "s/.*INFO.*}  //" | tr "\n" " ")"
  done
done
echo "=== all done ($(date "+%F %T"))"; touch runs/CL_DONE
