#!/usr/bin/env bash
# S1-EP: sidecar -> smoke training -> 3-seed training -> 3-seed closed loop. Stops at the first failure.
set -u
cd ~/IDL_project/dps-true-e
eval "$(~/miniforge3/bin/conda shell.bash hook)"; conda activate diffusion_planner; source setup/config.env
export PYTHONPATH=$PWD
LABELS=~/nuplan/labels/train-labels-2k
SIDE=~/nuplan/labels/train-labels-2k-epcv
SIMLOG=$NUPLAN_EXP_ROOT/exp/simulation/closed_loop_nonreactive_agents/dp_k16_label_noise1.5/train-labels-2k/2026-11-01-21-02-35/simulation_log
TRAIN=(--data $LABELS --labels cv --split-seed 0 --arch timestep --epochs 30 --step-lane scalars
       --progress-mode ep_cv --sidecar $SIDE --progress-weight 0)
step() { echo "=== $1 ($(date "+%F %T"))"; }

step "sidecar"
python scripts/make_epcv_sidecar.py --labels $LABELS --simlog $SIMLOG --out $SIDE --workers 6 || { echo "sidecar failed"; exit 1; }
echo "    sidecar files: $(find $SIDE -name '*.npz' | wc -l) of $(find $LABELS -name '*.npz' | wc -l) labels"

step "smoke training (1 epoch, 5% of train scenarios)"
python -m diffusion_planner.scorer.train "${TRAIN[@]}" --epochs 1 --train-frac 0.05 --out runs/s1_ep_cv_smoke \
  > runs/s1_ep_cv_smoke.log 2>&1 || { echo "smoke training failed"; tail -20 runs/s1_ep_cv_smoke.log; exit 1; }
grep -E "^\[train\]|^\[epoch" runs/s1_ep_cv_smoke.log | tail -4

for s in 0 1 2; do
  step "train seed $s"
  python -m diffusion_planner.scorer.train "${TRAIN[@]}" --seed $s --out runs/s1_ep_cv/seed$s \
    > runs/s1_ep_cv_seed$s.log 2>&1 || { echo "training seed $s failed"; tail -20 runs/s1_ep_cv_seed$s.log; exit 1; }
  grep -E "^\[epoch" runs/s1_ep_cv_seed$s.log | tail -1
done
touch runs/S1EP_TRAIN_DONE

for s in 0 1 2; do
  step "closed loop seed $s"
  K=16 NOISE=1.5 SEED=$s SELECTOR=verifier_ep_cv VERIFIER_TAG=s1ep VERIFIER_RAW=computed \
    VERIFIER_CKPT=runs/s1_ep_cv/seed$s/best.pt \
    bash scripts/run_sim.sh dp_k test14-hard-100 nr "~callback.simulation_log_callback" > runs/s1ep_cl_seed$s.log 2>&1
  echo "    exit $? | $(grep -hE "Number of (successful|failed) simulations" runs/s1ep_cl_seed$s.log | sed "s/.*INFO.*}  //" | tr "\n" " ")"
done
step "all done"; touch runs/S1EP_DONE
