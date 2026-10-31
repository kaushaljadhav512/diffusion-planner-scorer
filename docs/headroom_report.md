# Headroom report: best-of-K selection for Diffusion Planner (weeks 1–2)

Results of the headroom study on the 100-scenario Test14-hard subset: does sampling K plans from Diffusion Planner
(DP) and picking one improve closed-loop score (CLS), and which candidate set and rule-based selector should the
learned verifier be trained against?

Data: `results/headroom_test14-hard-100.csv` (98 runs, 0 failed), `results/headroom_test14-hard-100_seeds.csv`,
`results/per_scenario_test14-hard-100.csv`, `results/runtime_bench.csv`, `results/runtime_k_sweep.csv`,
`results/sample_diversity.md`. Numbers are means over seeds 0–2 unless marked otherwise.

## 1. Summary

- **The headroom is real.** Among K=16 candidates there are plans that score higher than DP's own plan. The best
  configuration, `pdm_sl` selection at noise 1.5, gains **+2.31 CLS averaged over NR and R** over vanilla DP
  (78.52 NR / 74.18 R vs 77.43 / 70.66).
- **Noise is a trade-off.** Raising the sampling noise from 0.5 to 1.5 creates diverse candidates but lowers the
  quality of candidate 0 in NR (−2.3 CLS). Selection at high noise first has to win that back.
- **PDM selection has a speed-limit blind spot.** PDM's score has no speed-limit term, and every PDM-selected run
  lost 2–3 points of speed-limit compliance. Adding nuPlan's speed-limit metric (`pdm_sl`) recovers it fully and
  makes `pdm_sl` the best selector in every setting, above even the oracle in NR.
- **Better forecasts add little.** The oracle (PDM with logged agent futures) is within seed noise of PDM, so
  constant-velocity forecasting is not what limits rule-based selection.
- **PDM's cost grows linearly with K; the verifier's does not.** PDM selection takes 18.5 ms at K=4 and 113.5 ms at
  K=64. The verifier takes ~1.1 ms at any K when it reuses DP's encoding.
- **Recommendation (task 2.4):** label with candidates at **noise 1.5**, driven by **`pdm_sl`**. Mixed noise
  (candidate 0 at 0.5, the rest at 1.5) does keep vanilla's NR score, but its NR gains over `first` stay under the
  seed-noise threshold, so by the plan's rule it does not qualify.

## 2. Setup

| | |
|---|---|
| Planner | Diffusion Planner, released checkpoint, no refinement; 10 denoising steps |
| Candidates | K=16 per planning step from one encoder pass (`sample_candidates`), output `[B, K, 11, 80, 4]` |
| Benchmark | `test14-hard-100`: 100 Test14-hard scenarios, balanced over the 14 scenario types |
| Challenges | NR (non-reactive agents) and R (reactive agents) |
| Seeds | 0, 1, 2 for every configuration, except `pdm_sl` at noise 0.5 (seed 0 only) |
| Hardware | RTX 5080 (16 GB) |

**Selectors**
- `first`: executes candidate 0. With noise 0.5 this is exactly vanilla DP.
- `random`: a uniformly random candidate.
- `pdm`: PDM-Closed scoring of each candidate with constant-velocity forecasts of the other agents (rule-based
  baseline).
- `pdm_sl`: `pdm` plus nuPlan's `speed_limit_compliance` metric, combined with CLS's weights (5/5/4/2).
- `oracle`: PDM scoring with the agents' logged futures. NR only, since logged futures are exact only there.

**Noise settings:** 0.5 (DP's default), 1.0, 1.5, and mixed `0.5-1.5` (candidate 0 at 0.5, candidates 1–15 at
1.5). Candidate noise is drawn candidate-major, so under mixed noise candidate 0 receives vanilla DP's exact draw.

**Reference:** on the full Test14-hard (272 scenarios), the same checkpoint scores 74.63 NR / 68.63 R (paper:
75.99 / 69.22). The 100-scenario subset is easier: vanilla scores 77.43 / 70.66 there.

**Seed noise:** the CLS std of `first` over three seeds is 0.4–0.8. A difference under ~1.5 CLS between two 3-seed
means is treated as noise.

## 3. Week 1: is there headroom?

### 3.1 Sample diversity

Ego-plan spread at K=16 over 100 test scenes (metres; `results/sample_diversity.md`):

| noise | endpoint spread | lateral std | pairwise ADE | FDE of candidate 0 | min FDE over K |
|---|---|---|---|---|---|
| 0.5 | 0.49 | 0.14 | 0.24 | 5.55 | 4.65 |
| 1.0 | 1.09 | 0.31 | 0.51 | 5.63 | 3.85 |
| 1.5 | 1.83 | 0.49 | 0.80 | 5.84 | 3.32 |

At DP's default noise the 16 candidates are nearly identical (endpoints within half a metre), so selection has
little to choose from. Spread grows roughly linearly with noise. Candidate 0 gets slightly worse (+0.3 m FDE), while
the best candidate gets much closer to the expert (−1.3 m min FDE).

### 3.2 Headroom grid (noise 0.5 / 1.0 / 1.5)

CLS, mean of 3 seeds; Δ is against `first` at the same noise.

| selector | NR 0.5 | NR 1.0 | NR 1.5 | R 0.5 | R 1.0 | R 1.5 |
|---|---|---|---|---|---|---|
| `first` | **77.43** | 76.00 | 75.13 | 70.66 | 72.37 | 73.00 |
| `random` | 77.89 (+0.46) | 76.52 (+0.52) | 75.90 (+0.78) | 70.16 (−0.50) | 72.31 (−0.06) | 73.57 (+0.56) |
| `pdm` | 77.55 (+0.12) | 76.16 (+0.16) | 77.87 (**+2.74**) | 71.38 (+0.72) | 72.05 (−0.32) | 73.02 (+0.01) |
| `oracle` | 77.39 (−0.04) | 76.42 (+0.42) | 78.07 (**+2.94**) | — | — | — |

- **NR:** at noise 0.5 no selector beats `first`, since the candidates barely differ. At noise 1.5, `pdm` and
  `oracle` gain about 2.8 over `first`, but `first` itself has dropped 2.3 points, so the net gain over vanilla DP
  is only +0.4–0.6.
- **R:** noise helps on its own (`first` +2.3 from 0.5 to 1.5), and selection adds nothing on top.
- **Noise 1.0 is worse than both 0.5 and 1.5** for every selector in NR. With 3 seeds this is a consistent but
  unexplained dip. It does not change any decision, since neither choice uses noise 1.0.
- `oracle` ≈ `pdm` (+0.20 at noise 1.5): logged futures barely beat constant-velocity forecasts.

### 3.3 Where PDM selection helps and hurts (noise 1.5, `pdm` − `first`)

| metric | NR | R |
|---|---|---|
| CLS | **+2.74** | +0.01 |
| no at-fault collisions | +3.0 | −1.2 |
| time-to-collision within bound | +5.3 | +0.3 |
| progress along expert route | +0.9 | **+3.9** |
| comfort | +4.0 | −0.3 |
| speed-limit compliance | **−2.7** | **−2.3** |

In NR, selection helps by avoiding collisions and TTC violations. In R, it improves progress but loses it all back to
speed-limit violations and to collisions that constant-velocity forecasts miss. The speed-limit loss shows up in
every PDM-scored configuration. PDM's score has no speed-limit term, but CLS weights it 4 of 16 in the weighted part,
which costs about 0.5 CLS.

**Week 1 decision (task 1.11): conditional go.** The candidates contain better plans, but at high noise selection
mostly repairs damage the noise did. Two fixes were tested in week 2: mixed noise, to keep vanilla's plan in the
candidate set, and a speed-limit term for PDM.

## 4. Week 2: mixed noise and `pdm_sl`

### 4.1 Mixed-noise grid

CLS, mean of 3 seeds; Δ is against `first` at the same noise setting.

| selector | NR mixed | NR 1.5 | R mixed | R 1.5 |
|---|---|---|---|---|
| `first` | 77.43 | 75.13 | 70.66 | 73.00 |
| `random` | 75.92 (−1.51) | 75.90 (+0.78) | 73.26 (+2.61) | 73.57 (+0.56) |
| `pdm` | 78.12 (+0.69) | 77.87 (+2.74) | 72.68 (+2.02) | 73.02 (+0.01) |
| `pdm_sl` | 78.37 (+0.94) | **78.52 (+3.39)** | 73.84 (+3.18) | **74.18 (+1.18)** |
| `oracle` | 78.24 (+0.81) | 78.07 (+2.94) | — | — |

- **Mixed noise does what it was designed to do for `first`:** its CLS matches vanilla DP to every digit in both NR
  and R, which confirms that candidate 0 is vanilla's plan. The 5-scenario smoke check gave the same match (72.57 vs
  72.57).
- **In NR, mixed noise ends up in the same place as noise 1.5.** Selectors reach 78.1–78.4 with mixed noise and
  77.9–78.5 at noise 1.5. The selection gain over `first` looks smaller under mixed noise only because its `first` is
  the stronger vanilla plan.
- **In R, both settings exceed the high-noise level** (~73), with `pdm_sl` at 73.8 (mixed) and 74.2 (1.5).
- `random` under mixed noise picks a 1.5-noise candidate 15 times out of 16, so it behaves like `random` at 1.5 in
  NR. This works as a sanity check.

### 4.2 `pdm_sl` closes the speed-limit gap

Speed-limit compliance (%) and CLS, mean of 3 seeds:

| config | NR speed limit | NR CLS | R speed limit | R CLS |
|---|---|---|---|---|
| `first`, noise 0.5 (vanilla) | 95.54 | 77.43 | 97.17 | 70.66 |
| `pdm`, noise 1.5 | 92.62 | 77.87 | 94.65 | 73.02 |
| `pdm_sl`, noise 1.5 | **96.62** | **78.52** | **98.04** | **74.18** |
| `oracle`, noise 1.5 | 92.56 | 78.07 | — | — |

`pdm_sl` recovers the full 3–4 points PDM lost and ends above vanilla on speed limit. The other metrics stay at
PDM's level (collisions 89.5 vs 89.5 in NR; TTC 78.7 vs 79.3). It beats `pdm` by +0.65 NR and +1.16 R, and in NR
it beats the oracle, which shares PDM's blind spot. It also fixes most of PDM's weakness in R (Δ vs `first` +0.01
→ +1.18).

### 4.3 Overall ranking

NR+R average CLS (oracle excluded, since it has no R result):

| config | NR | R | avg | vs vanilla |
|---|---|---|---|---|
| vanilla (`first`, 0.5) | 77.43 | 70.66 | 74.04 | — |
| `pdm`, noise 1.5 (week 1 best) | 77.87 | 73.02 | 75.45 | +1.41 |
| `pdm`, mixed | 78.12 | 72.68 | 75.40 | +1.36 |
| `pdm_sl`, mixed | 78.37 | 73.84 | 76.10 | +2.06 |
| **`pdm_sl`, noise 1.5** | **78.52** | **74.18** | **76.35** | **+2.31** |

## 5. Per-scenario view

From `results/per_scenario_test14-hard-100.csv`. "Consistent" means the scenario moved the same way against `first`
in all three seeds; "insensitive" means no selector changes its score at any seed.

| config | challenge | consistent gains | consistent losses | insensitive |
|---|---|---|---|---|
| `pdm`, noise 1.5 | NR | 36 | 16 | 34 |
| `pdm_sl`, noise 1.5 | NR | 42 | 8 | 34 |
| `pdm_sl`, mixed | NR | 42 | 10 | 34 |
| `oracle`, noise 1.5 | NR | 40 | 16 | 34 |
| `pdm`, noise 1.5 | R | 35 | 14 | 42 |
| `pdm_sl`, noise 1.5 | R | 44 | 7 | 42 |
| `pdm_sl`, mixed | R | 42 | 5 | 43 |

- About a third of the scenarios (34 in NR, 42–43 in R) are insensitive: every candidate leads to the same outcome.
  Any selector, including the verifier, can only gain on the remaining two-thirds.
- `pdm_sl` halves the consistent losses compared with `pdm` (16 → 8 in NR, 14 → 7 in R) while also adding gains.
  That is a cleaner improvement than the CLS means alone show.

**By scenario type** (`pdm_sl` at noise 1.5 vs `first`; about 7 scenarios per type, so treat these as indicative):

- *Largest gains, NR:* traversing pickup/drop-off (+15.3), waiting for a pedestrian to cross (+14.1), starting a
  straight traffic-light intersection traversal (+11.0), following a lane with a lead (+6.1).
- *Largest gains, R:* following a lane with a lead (+11.3), high-magnitude speed (+8.8), changing lane (+5.5).
- *Consistent losses:* behind a long vehicle (−7.4 NR, −14.3 R). Changing lane is −7.9 in NR but +5.5 in R.
  The oracle shows the same NR losses on these two types, so they reflect PDM's scoring, not its forecasts. These
  are good candidates for failure-case analysis (task 4.4).

## 6. Runtime

**Single process, no contention** (`results/runtime_bench.csv`, ms):

| K | DP encoder | sampling (10 steps) | verifier (reuses encoding) | verifier (own encoder) |
|---|---|---|---|---|
| 4 | 2.5 | 25.5 | 1.07 | 3.37 |
| 16 | 2.4 | 26.4 | 1.10 | 3.35 |
| 64 | 2.4 | 30.2 | 1.24 | 3.55 |

**One simulation worker, 10 scenarios, noise 1.5** (`results/runtime_k_sweep.csv`, ms per planning step):

| K | step, `first` | step, `pdm` | PDM selection |
|---|---|---|---|
| 4 | 114.0 | 131.2 | 18.5 |
| 8 | 112.7 | 136.2 | 23.5 |
| 16 | 114.0 | 149.8 | 35.7 |
| 32 | 114.3 | 176.8 | 62.8 |
| 64 | 118.0 | 230.9 | 113.5 |

- Batched sampling is nearly free in K: 25.5 → 30.2 ms from K=4 to 64.
- PDM selection grows roughly linearly with K, at about 1.6 ms per candidate plus a fixed ~12 ms. At K=64 it takes
  half of the planning step.
- The (untrained) verifier architecture costs ~1.1 ms at any K when it reuses the sampler's DP encoding, which is
  **~32× cheaper than PDM at K=16 and ~90× at K=64**. Task 3.4 should pass the encoding through.
- The in-grid timings (`step_ms` ≈ 200–220, `select_ms` ≈ 42–44 for PDM, ≈ 206 for the oracle) come from parallel
  workers sharing the GPU and are too noisy for the final runtime table.
- The CLS columns of `runtime_k_sweep.csv` come from 10 scenarios with one seed and are not meaningful as scores.

## 7. Decision (task 2.4)

**Candidate set: noise 1.5, K=16. Label-driving selector: `pdm_sl`.**

- The plan's rule was "mixed if the mixed grid hits the target, otherwise noise 1.5". Mixed noise meets the NR
  vanilla level (77.43) and the R target (~73), but its NR gains over `first` (`pdm_sl` +0.94, `oracle` +0.81,
  `pdm` +0.69) are all below the ~1.5 seed-noise threshold. It does not qualify.
- Noise 1.5 also has the best absolute scores (+0.15 NR, +0.34 R over mixed). That gap is within seed noise, so the
  choice is close, and either set would support the verifier study.
- Noise 1.5 gives the verifier more diverse candidates to rank (16 high-noise plans rather than 15), which is what
  the labels need to be informative.
- `pdm_sl` replaces `pdm` as the rule-based baseline the verifier must match: it is the strongest selector and
  removes the speed-limit blind spot the verifier would otherwise learn from PDM-style labels.

**Target for the verifier (task 3.5):** match `pdm_sl` at noise 1.5 (+2.31 NR+R over vanilla) at a fraction of its
cost. A verifier at oracle-level accuracy cannot be expected to beat `pdm_sl`, since the oracle is not above it.

## 8. Caveats

- **Subset, not the full benchmark.** All numbers come from the 100-scenario subset, which is easier than the full
  Test14-hard (vanilla 77.4 vs 74.6 NR). Full-benchmark runs are task 4.1.
- **Three seeds.** Differences under ~1.5 CLS are within seed noise. This covers the mixed-vs-1.5 comparison and
  `pdm` vs `oracle`.
- **`pdm_sl` at noise 0.5** has one seed only.
- **Smoke runs are not results.** The `_smoke5` and `_smoke10` CSVs check that the runner works. With 5 scenarios,
  one scenario flips the CLS by ±20.
- **Per-type numbers** rest on about 7 scenarios per type.
- **The noise-1.0 dip** in NR is unexplained.

## 9. Next steps

1. Record the 2.4 decision in the project plan.
2. Download one nuPlan train city archive to the GPU machine (tasks 1.1 / 2.5). No train or mini data is there yet,
   and the disk has ~150 GB free.
3. Build the `train-labels-2k` filter, run the 5-scenario smoke label run, then the full label run (~4 h, ~60k
   samples) with noise 1.5 and `pdm_sl` (tasks 2.6–2.8).
4. Label statistics (2.9): check the informative-sample rate. Since about a third of the test scenarios are
   insensitive, expect a similar share of uninformative training samples.
5. Overfit check on real labels (2.12), then full training (week 3).
6. Confirm `test_mixed_noise_candidate0_matches_vanilla` with the real checkpoint (2.1). The closed-loop match in
   4.1 already shows the property holds.

## Reproduce

```bash
source setup/config.env
# week 1 grid (noise 0.5 / 1.0 / 1.5, seeds 0-2)
for n in 0.5 1.0 1.5; do for s in 0 1 2; do NOISE=$n SEED=$s bash scripts/run_headroom.sh; done; done
# week 2: mixed-noise grid and pdm_sl at 1.5
for s in 0 1 2; do NOISE=0.5,1.5 SEED=$s bash scripts/run_headroom.sh; done
for s in 0 1 2; do NOISE=1.5 SEED=$s SELECTORS="first pdm_sl" bash scripts/run_headroom.sh; done
# per-scenario comparison and runtime
python scripts/compare_scenarios.py --split test14-hard-100 --config dp_k16_pdm_sl_noise0.5-1.5 --challenge nr
python scripts/bench_runtime.py
bash scripts/run_runtime_sweep.sh
```
Run guides: [week1_evals.md](week1_evals.md), [week1_noise_sweep.md](week1_noise_sweep.md),
[week2_runs.md](week2_runs.md).
