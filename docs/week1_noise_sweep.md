# Week 1 follow-up: seeds and noise sweep

Instructions for re-running the headroom grid with several seeds and higher sampling noise, on the same GPU
machine as the [week 1 evals](week1_evals.md). Expect about 30 minutes of hands-on work and one long overnight
run (about 7 hours).

## Why

The week 1 headroom run (K=16, noise 0.5, seed 0) gave:

| NR, gain over `first` | CLS |
|---|---|
| `oracle` | +0.44 |
| `pdm` | +0.64 |
| `random` | +0.55 |

The oracle is under the 1-point go bar, and `random` gains as much as the oracle, so we can't tell any of these
apart from seed noise. The sample diversity table explains the low headroom: at noise 0.5 the 16 candidates end
a median of only 0.26 m apart after 8 s. At noise 1.5 the best-of-16 endpoint error is 1.3 m better than at
0.5.

This run answers two questions:
1. **How big is seed noise?** Same grid, seeds 0, 1, 2.
2. **Does higher noise create headroom?** The same at noise 1.0 and 1.5.

## What's new in the scripts

- **`SEED=n`** (default 0) sets the planner's torch seed. Runs with a non-zero seed are named
  `..._seed<n>`, e.g. `dp_k16_oracle_noise1.0_seed2`. Seed-0 runs keep their old names, so last week's runs
  count as seed 0.
- **`vs_first`** is now the gain over `first` at the same K, noise **and seed**.
- **A new table, "over seeds"**, gives each configuration's mean and std of CLS and of `vs_first`. It is
  printed after the usual tables and saved to `results/headroom_test14-hard-100_seeds.csv`.

Start every shell with:
```bash
cd ~/diffusion-planner-scorer            # wherever the repo is
conda activate diffusion_planner
source setup/config.env
git pull
```

---

## 1. Check the update (~10 min)

- [ ] `pytest tests/test_summarize_results.py`: all 5 tests pass.
- [ ] Smoke-test a seeded run:
  ```bash
  SMOKE=5 SEED=1 SELECTORS="first oracle" CHALLENGES=nr bash scripts/run_headroom.sh
  ```
  **Done when:**
  - the table lists `dp_k16_first_seed1` and `dp_k16_oracle_seed1` with `n` = 5 and `failed` = 0;
  - `dp_k16_first_seed1` has a slightly different CLS from last week's `dp_k16_first` smoke run (if you still
    have it). The same CLS would mean the seed isn't reaching the planner; stop and report it.

## 2. Full sweep (overnight, ~7 h)

8 grids of 7 runs each, about 50 minutes per grid. The order matters: seed noise at 0.5 comes first because
it's the cheapest and everything else is judged against it.

| Grids | Noise | Seeds | Time |
|---|---|---|---|
| 2 | 0.5 | 1, 2 (seed 0 is last week's run) | ~1.7 h |
| 3 | 1.0 | 0, 1, 2 | ~2.5 h |
| 3 | 1.5 | 0, 1, 2 | ~2.5 h |

- [ ] Start it **inside tmux**:
  ```bash
  tmux new -s sweep
  for cfg in "0.5 1" "0.5 2" "1.0 0" "1.0 1" "1.0 2" "1.5 0" "1.5 1" "1.5 2"; do
    set -- $cfg
    NOISE=$1 SEED=$2 bash scripts/run_headroom.sh
  done 2>&1 | tee "$NUPLAN_EXP_ROOT/logs/sweep_$(date +%m%d-%H%M).log"
  ```
  Detach with `Ctrl-b d`; check later with `tmux attach -t sweep`.
- [ ] Next morning, check that every grid ended without `Failed runs:`. Then print the final tables:
  ```bash
  grep -E "^=== K=|ok in|FAILED|Failed runs" "$NUPLAN_EXP_ROOT"/logs/sweep_*.log | tail -60
  python scripts/summarize_results.py --split test14-hard-100 \
    --csv results/headroom_test14-hard-100.csv --seeds-csv results/headroom_test14-hard-100_seeds.csv
  ```
  **Done when:** the "over seeds" table has `seeds` = 3 for every row: 7 configurations × 3 noise levels.
- [ ] If a run failed, re-run only that one, then re-print the tables with the command above:
  ```bash
  NOISE=1.0 SEED=2 SELECTORS=pdm CHALLENGES=r bash scripts/run_headroom.sh    # example
  ```

**Don't change** K (16) or the split (`test14-hard-100`).

## 3. Read the results

Use the **over seeds** table. Rows are named by configuration without the seed, e.g. `dp_k16_oracle_noise1.0`.
Noise-0.5 rows have no `_noise` suffix.

| Read | Where | What it tells us |
|---|---|---|
| **Seed noise** | `CLS_std` of the `dp_k16_first*` rows | How much CLS moves from seed alone. A gain under about 2× this is noise. |
| **Headroom** | `vs_first_mean` ± `vs_first_std`, `oracle`, NR, at each noise | The go/no-go number. |
| Cost of noise | `CLS_mean` of `first` at 1.0 and 1.5 vs 0.5 | Whether higher noise makes vanilla DP itself worse. A drop eats into any selection gain. |
| Realistic selector | `pdm`, NR and R | Whether last week's +1.70 in R holds up across seeds. |
| Sanity check | `random` | Should average about 0. |

**Decision rule** (at the best noise level):
- **Oracle gain ≥ 1 CLS point and > 2× the seed std, and `first` not much worse → go.** Week 2 builds the
  learned verifier on DP samples at that noise.
- **Under that bar even at 1.5 → mixed candidate set:** add PDM-Closed's 15 centreline proposals (5 target
  speeds × 3 lateral offsets) to the 16 DP samples, then rerun the oracle. That needs a code change first.
- **Still nothing with mixed candidates →** pivot to failure prediction.

## 4. Report back

- [ ] Commit and push the tables:
  ```bash
  git add results/headroom_test14-hard-100.csv results/headroom_test14-hard-100_seeds.csv
  git commit -m "Seed and noise sweep results (noise 0.5 / 1.0 / 1.5, seeds 0-2)" && git push
  ```
- [ ] Post to the team:
  - seed noise: `CLS_std` of `first` at noise 0.5, NR and R;
  - the oracle's `vs_first_mean` ± std at each noise (NR);
  - `first`'s `CLS_mean` at each noise;
  - `pdm`'s `vs_first_mean` ± std in NR and R at each noise;
  - total wall-clock time, and anything that failed, with the tail of its log.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| Out of memory, machine freezes | Put `THREADS=6` in front of `bash scripts/run_headroom.sh` inside the loop. |
| Sweep won't fit in one night | Drop R: add `CHALLENGES=nr` inside the loop (NR holds the go/no-go number). Or run noise 0.5 + 1.0 tonight and 1.5 tomorrow. |
| "over seeds" table missing | It's only printed once some configuration has more than one seed. Check that `SEED` was set: run names should end in `_seed1`, `_seed2`. |
| `vs_first` empty for a seeded run | That seed's `first` run is missing or failed; re-run `SELECTORS=first` with the same `NOISE` and `SEED`. |
| Older runs show up | Only the latest run per name is used. `--all` lists every run, including old ones. |
