# Running the week 1 evals

Instructions for one person running all week 1 evaluations on one GPU machine. Expect about an hour of
hands-on work and one overnight run. The machine must already be set up and verified with
[getting_started.md](getting_started.md): `pytest` passing with no skips, and a `SMOKE=5` run working.

## What this measures

**Can picking among Diffusion Planner's samples beat vanilla DP?** At each step DP samples K=16 candidate
plans and a selector picks one. Four selectors run on a fixed 100-scenario subset of Test14-hard:

| Selector | What it is |
|---|---|
| `first` | Always candidate 0, i.e. vanilla DP. The baseline. |
| `random` | A random candidate. |
| `pdm` | PDM's rule-based scorer picks the best candidate. A realistic selector. |
| `oracle` | PDM's scorer, but it knows where the other cars actually go. The upper bound. Non-reactive (NR) only. |

The deciding number is **the oracle's CLS gain over `first` in NR**: the headroom a better selector could
capture. The runs also record selection time per step, which a learned scorer will later need to beat.

Because everything runs on one machine, the scores and timings are directly comparable.

Start every shell with:
```bash
cd ~/diffusion-planner-scorer            # wherever the repo is
conda activate diffusion_planner
source setup/config.env
git pull
```

---

## 1. Build the 100-scenario subset (~5 min)

- [ ] `python scripts/make_subset_filter.py`
  - Prints a per-type table ending in `picked 100`, with 7–8 scenarios per type.
  - Writes `diffusion_planner/config/scenario_filter/test14-hard-100.yaml`.
  - If it reports missing tokens, the test split download is incomplete; re-run `setup/download_nuplan.sh`.
- [ ] Check nuPlan reads it: `SMOKE=5 bash scripts/run_sim.sh dp test14-hard-100 nr`
- [ ] Commit and push it, so later runs (and teammates) use the same 100 scenarios:
  ```bash
  git add diffusion_planner/config/scenario_filter/test14-hard-100.yaml
  git commit -m "Add the 100-scenario Test14-hard subset filter" && git push
  ```

## 2. Smoke-test every selector (~30–60 min)

5 scenarios per selector (NR and R, i.e. all 7 runs at small scale), to catch problems before the long run:
- [ ] `SMOKE=5 bash scripts/run_headroom.sh` (in R it skips `oracle` automatically; that's expected)
- [ ] Check the table at the end:
  - all runs listed, `n` = 5, `failed` = 0, and no `FAILED` lines above the table;
  - `sample_ms`, `select_ms` and `vs_first` columns are filled in;
  - `first`'s CLS is in the same range as plain `dp` on the same scenarios. Check with
    `SMOKE=5 bash scripts/run_sim.sh dp test14-hard-100 nr`; it won't match exactly, since the random noise
    differs.
- [ ] **Estimate the full run.** Each run prints `ok in N min`. The full run takes roughly
  (sum of those times) × 20. If that's more than one night, reduce scope (see "If something goes wrong").
- [ ] Note `select_ms` for `pdm` vs `oracle`. The oracle reads 40 future frames from the DB every step; if it
  is many times slower than `pdm`, mention it when reporting.

A failed run prints the tail of its log; full logs are in `$NUPLAN_EXP_ROOT/logs/`.

## 3. Sample diversity (~15–20 min, task 1.4)

Can run while you check the smoke results; it uses the GPU only briefly.
- [ ] `python scripts/sample_diversity.py`. The first run caches 100 test scenes, which is the slow part.
- [ ] Check `results/sample_diversity.md`: one row per noise scale (0.5, 1.0, 1.5). As noise grows,
  `endpoint_spread` and `pairwise_ade` should grow too. The gap between `fde_first` and `min_fde` is an early
  hint of headroom.

## 4. Full headroom run (several hours, overnight)

- [ ] Start it **inside tmux**, so it survives closing the terminal:
  ```bash
  tmux new -s sim
  bash scripts/run_headroom.sh
  ```
  Detach with `Ctrl-b d`; check on it later with `tmux attach -t sim`.
- [ ] It runs 7 simulations of 100 scenarios, one after another:
  - NR: `first`, `random`, `pdm`, `oracle`
  - R: `first`, `random`, `pdm`

  A failed run doesn't stop the rest.
- [ ] Next morning, check the final table: 7 rows with `n` = 100 and `failed` = 0. It's saved to
  `results/headroom_test14-hard-100.csv`.
- [ ] If a run failed, re-run just that one, then re-print the table:
  ```bash
  SELECTORS=pdm CHALLENGES=r bash scripts/run_headroom.sh          # example: only R / pdm
  python scripts/summarize_results.py --split test14-hard-100 --csv results/headroom_test14-hard-100.csv
  ```

**Don't change** `K`, `NOISE` or the split for this run: K=16, noise 0.5, `test14-hard-100`.

## 5. Read the results and decide (task 1.11)

From the best-of-K table (`vs_first` = CLS minus `first`'s CLS, same challenge):

| Read | Where | What to expect |
|---|---|---|
| **Headroom** | NR, `oracle`, `vs_first` | The go/no-go number. |
| Realistic selector | NR and R, `pdm`, `vs_first` | How much a hand-made scorer already gets. |
| Sanity check | NR and R, `random`, `vs_first` | About 0 or negative. If random clearly beats `first`, something is wrong. |
| Cost | `select_ms` (`pdm`, `oracle`), `sample_ms` | Time per step the learned scorer is compared against. |

**Decision:**
- **Oracle gain ≥ 1 CLS point → go.** Week 2 builds the learned scorer.
- **Under 1 point → make the candidates more varied and rerun:** `NOISE=1.0 bash scripts/run_headroom.sh`
  (another night; `NOISE=1.5` if still low). The diversity table from step 3 suggests which noise level is
  worth trying. These runs are stored separately and appear as `_noise1.0` rows in the same table, with
  their own `vs_first`.

## 6. Report back

- [ ] Commit and push the results:
  ```bash
  git add results/headroom_test14-hard-100.csv results/sample_diversity.md results/sample_diversity.csv
  git commit -m "Week 1 headroom and sample diversity results" && git push
  ```
- [ ] Post to the team:
  - the headroom number and the decision;
  - the `pdm` and `random` gains;
  - `select_ms` / `sample_ms`;
  - the machine used (GPU, RAM, `threads=` from the first output line) and how long the full run took;
  - anything that failed, with the tail of its log.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| Machine freezes, run killed, out of memory | Fewer parallel simulations: put `THREADS=6` in front of the command (default is RAM ÷ 3 GB). |
| `CUDA out of memory` | `GPU_FRAC=0.1 THREADS=6 bash scripts/run_headroom.sh` |
| Full run would take more than a night | Run NR first, since it holds the go/no-go number: `CHALLENGES=nr bash scripts/run_headroom.sh`. Do R the next night with `CHALLENGES=r`. |
| `no scenario filter test14-hard-100` | Step 1 not done, or not pulled. |
| `oracle` refused with `r` | Expected: logged futures are only exact for non-reactive agents. |
| A run says `FAILED` | Read the end of its log in `$NUPLAN_EXP_ROOT/logs/`, then re-run only that selector and challenge (step 4). |
| Results table missing a run | `python scripts/summarize_results.py --split test14-hard-100 --all` lists every run, including older ones. |
