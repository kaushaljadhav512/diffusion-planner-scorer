# Expert-relative progress labels: oracle selectors in closed loop

Does replacing PDM's progress (relative to the best of the K candidates) with nuPlan's definition (relative to the
human driver's logged progress, plus `ego_is_making_progress`) give a label that tracks the closed-loop score better?
Both are tested as oracle selectors (logged agent futures), so the only difference is the label definition.

Setup: Test14-hard 100-scenario subset, non-reactive, K = 16, noise 1.5, seeds 0-2 (`scripts/run_headroom.sh`).
Tables: `results/headroom_test14-hard-100{,_seeds}.csv`.

| Selector | Label definition | CLS (mean ± std, 3 seeds) |
|---|---|---|
| First | none | 75.13 ± 0.68 |
| `oracle` | PDM's own score (no speed limit) | 78.07 ± 0.59 |
| `pdm_sl` | constant-velocity forecasts, CLS weights | 78.52 ± 0.51 |
| `oracle_sl` | **current training target** (logged futures, CLS weights) | 78.76 ± 0.57 |
| `oracle_ep` | **expert-relative progress + making progress** | **80.14 ± 0.35** |

- `oracle_ep` beats `oracle_sl` on every seed: +0.98, +2.41, +0.73 (mean +1.37). The mean is just under the ~1.5 CLS
  seed-noise threshold used in the progress report, but all three paired differences are positive.
- The gain does not come from more progress (route progress stays at about 94% for both). It comes from time-to-collision
  (78-79% to 81-82% of scenarios) and comfort (91-92% to 93-94%): capping progress at the human driver's removes the
  incentive to pick faster, riskier candidates.
- The new definition changes the pick on 41% of steps (18,344 of 44,742; per-step logs in
  `$NUPLAN_HOME/exp/oracle_ep_stats/` on the GPU machine).

Caveat: these are oracle selectors (logged futures and the human driver's progress are only known offline), so this
compares label definitions, not deployable selectors. To use it, regenerate labels (they now record
`progress_expert`) and retrain the verifier on `metrics.combined_score_expert`.
