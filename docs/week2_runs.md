# Week 2 runs: candidate set, labels, runtime

Instructions for the week-2 runs on the GPU machine. Expect about 1 hour of hands-on work, one overnight
headroom run (~5 h), and one label run (~4 h) once the train data is downloaded.

## Why

Week 1 (100-scenario Test14-hard subset, K=16, 3 seeds) showed:
- at noise 1.5, picking with PDM gains +2.7 CLS over `first` in NR, but `first` itself drops ~2.5 points vs noise
  0.5, so the net gain over vanilla DP is about zero;
- every PDM-selected run loses 2–3 points of **speed-limit compliance**: PDM's score has no speed-limit term,
  but CLS weights it 4/16.

This week's runs test the two fixes, then build the verifier's training labels:
1. **Mixed noise**: candidate 0 at 0.5 (exactly vanilla DP's plan) plus 15 candidates at 1.5. The goal is to keep
   vanilla's NR score and add the high-noise gains on top.
2. **`pdm_sl`**: PDM plus nuPlan's speed-limit metric, combined with CLS's weights.

## What's new in the scripts

- **`NOISE=0.5,1.5`** gives candidate 0 noise 0.5 and the others 1.5. Runs are named `..._noise0.5-1.5`.
- **`SELECTOR=pdm_sl`** (`PDMSpeedLimitSelector`). `run_headroom.sh` now runs it by default:
  `first random pdm pdm_sl oracle` in NR, and the same without `oracle` in R.
- **`SELECTOR=label`** (`LabelRecorder`, NR only) drives with `pdm_sl` and writes verifier labels to
  `LABEL_DIR` (default `$NUPLAN_HOME/labels/<split>`).
- New scripts: `make_train_filter.py`, `label_stats.py`, `bench_runtime.py`, `run_runtime_sweep.sh`,
  `compare_scenarios.py`; training: `python -m diffusion_planner.scorer.train`.

Start every shell with:
```bash
cd ~/diffusion-planner-scorer            # wherever the repo is
conda activate diffusion_planner
source setup/config.env
git pull
```

---

## 1. Check the update (~15 min)

- [ ] `pytest tests/`. **Done when:** everything passes. With the real checkpoint, `test_sampling.py::
  test_mixed_noise_candidate0_matches_vanilla` is the important one: candidate 0 of a mixed draw is vanilla DP's
  plan.
- [ ] Smoke test of mixed noise and `pdm_sl`:
  ```bash
  SMOKE=5 NOISE=0.5,1.5 SELECTORS="first pdm_sl" CHALLENGES=nr bash scripts/run_headroom.sh
  ```
  **Done when:** the table lists `dp_k16_first_noise0.5-1.5` and `dp_k16_pdm_sl_noise0.5-1.5` with `failed` = 0,
  and **`dp_k16_first_noise0.5-1.5` matches last week's seed-0 `dp_k16_first` smoke run (72.57) to within a few
  tenths**. `first` executes candidate 0, which gets exactly the noise it got at 0.5. (Over 5 scenarios a different
  seed moved the CLS by 24 points, so a near-match is a clear signal.) If it is far off, stop and report it.

## 2. Headroom grids (overnight, ~5 h; tasks 2.2, 2.3, 2.15)

One grid is 9 runs (~65 min). Run in tmux:
```bash
for seed in 0 1 2; do NOISE=0.5,1.5 SEED=$seed bash scripts/run_headroom.sh; done
for seed in 0 1 2; do NOISE=1.5 SEED=$seed SELECTORS="first pdm_sl" bash scripts/run_headroom.sh; done
NOISE=0.5 SEED=0 bash scripts/run_headroom.sh       # the missing seed-0 grid at 0.5
```
The second loop reruns `first` at noise 1.5 only so that `pdm_sl` gets a `vs_first` (same seed). Its CLS should
match last week's runs.

- [ ] **Done when** no grid printed `Failed runs:`. Commit `results/headroom_test14-hard-100{,_seeds}.csv`.
- [ ] Per-scenario comparison (task 2.14):
  ```bash
  python scripts/compare_scenarios.py --split test14-hard-100 --config dp_k16_pdm_sl_noise0.5-1.5 --challenge nr
  ```
  Commit `results/per_scenario_test14-hard-100.csv`.

**What to look for (decision 2.4):** in the seeds table, `dp_k16_first_noise0.5-1.5` in NR should stay near
`dp_k16_first` (77.7, vanilla's level). Mixed noise wins if, on top of that, `pdm` / `pdm_sl` / `oracle` are above
`first` in NR by more than seed noise (~1.5), and R stays near the high-noise level (~73). Otherwise use noise 1.5.
In the per-metric table, `pdm_sl` should recover the speed-limit points that `pdm` loses.

## 3. Runtime (~1 h; task 2.13)

- [ ] `python scripts/bench_runtime.py` (needs last week's `$NUPLAN_HOME/cache/test_diversity`). Writes
  `results/runtime_bench.csv`: encoder, sampling and verifier ms for K = 4…64, one process, no contention.
- [ ] `bash scripts/run_runtime_sweep.sh`: one-worker simulations of 10 scenarios, K = 4…64, `first` and `pdm`.
  Writes `results/runtime_k_sweep.csv`; the `select_ms` column is PDM's cost per K.
- [ ] Commit both CSVs.

## 4. Labels (tasks 2.5–2.9; needs a train city archive)

- [ ] Download one train city archive (e.g. Boston) with `setup/download_nuplan.sh` (it lands in
  `nuplan-v1.1/trainval/`).
- [ ] Build the filter: `python scripts/make_train_filter.py`. **Done when:** it prints ~140 per type and writes
  `diffusion_planner/config/scenario_filter/train-labels-2k.yaml`. Only logs of nuPlan's official train split are
  used. With one city some types may have fewer; that's fine. Commit the yaml.
- [ ] Smoke label run:
  ```bash
  SMOKE=5 SELECTOR=label NOISE=0.5,1.5 bash scripts/run_sim.sh dp_k train-labels-2k nr
  python scripts/label_stats.py $NUPLAN_HOME/labels/train-labels-2k
  ```
  **Done when:** the run has `failed` = 0, and `label_stats.py` reports ~150 samples from 5 scenarios, with
  `informative` > 0. Then delete `$NUPLAN_HOME/labels/train-labels-2k` before the full run.
- [ ] Full run (use the candidate set from step 2: drop `NOISE=0.5,1.5` if noise 1.5 won):
  ```bash
  SELECTOR=label NOISE=0.5,1.5 bash scripts/run_sim.sh dp_k train-labels-2k nr
  ```
  ~4 h, ~60k samples, ~4 GB. **Done when:** finished, then note samples, disk size and runtime in the tasks file.
- [ ] `python scripts/label_stats.py $NUPLAN_HOME/labels/train-labels-2k --csv results/label_stats.csv` and
  commit the CSV (task 2.9). The key numbers: `informative`, `headroom`, `cv_regret` (how much PDM selection
  leaves on the table offline), and `cv_agreement`.
- [x] Overfit check (task 2.12):
  ```bash
  python -m diffusion_planner.scorer.train --data $NUPLAN_HOME/labels/train-labels-2k --overfit 100 \
    --epochs 50 --lr 1e-3 --batch-size 16 --out runs/overfit
  ```
  **Done when:** `regret verifier` falls to ~0 by the last epochs. **Result:** regret 0.0004, top-1 0.80 (random:
  0.087 / 0.18) with the ranking loss (`--rank-weight 3 --rank-temperature 0.02`, now the defaults). BCE alone stalls at
  regret ~0.019 (soft targets, so loss floor ~0.09). The old command (batch 64, lr 3e-4) gives only 2 steps per epoch
  on 100 samples, so it reaches regret 0.044 in 50 epochs; use batch 16 for the overfit check.
