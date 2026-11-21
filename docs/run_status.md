# Learned verifier: status of runs

Updated 2026-11-20. The goal is a learned verifier that imitates PDM+SL's per-candidate scores (`metrics_cv`) so it
can pick among Diffusion Planner's 16 candidates much faster than PDM. All runs use `train-labels-2k`, cv labels,
split seed 0 (53,790 train / 5,970 val samples, 199 val scenarios), 30 epochs, 3 seeds. Lower regret is better;
random selection scores 0.0625 and PDM+SL 0.

## Runs

| Run | Branch / worktree | What | Regret | Top-1 | Status |
|---|---|---|---|---|---|
| w5_cv | `verifier-ablations` / `dps-verifier-ablations` | one token per plan, 7 metric heads | 0.0370 ± 0.0007 | 0.834 | done |
| w6_timestep | `timestep-verifier` / `dps-timestep-verifier` (commit `faf6319`) | one token per step (41 per candidate), 7 metric heads | **0.0352 ± 0.0013** | 0.819 | done |
| w7_timestep_classify | `timestep-classifier` / `dps-timestep-classifier` (uncommitted) | per-step model, one score per candidate, softmax over the 16 against PDM+SL's best (`--classify-target best`) | 0.0364 ± 0.0004 | **0.848** | done |

Run directories are under `runs/` in each worktree on aegis (`~/IDL_project/<worktree>/runs/<run>/seed{0,1,2}`),
each with `best.pt`, `log.jsonl` and `diagnostic_cv/`.

Diagnostics (branch `drivable-check` / `dps-drivable-check`, commit `d32fc96`):

| Script | Output on aegis |
|---|---|
| `scripts/oracle_substitution.py` (per-metric oracle substitution) | `dps-timestep-verifier/runs/w6_timestep/oracle_substitution_cv/` |
| `scripts/check_carpark_drivable.py` (PDM drivable test with lanes and car parks apart) | `dps-drivable-check/runs/carpark_check/` |

## What we learned

- **w6 vs w5:** per-step queries mainly improved collision ranking within a scene (pair accuracy 0.57 → 0.78) and
  TTC. Drivable-area detection improved across scenes (AUC 0.68 → 0.86) but not within a scene (pair accuracy
  0.58 → 0.57), so drivable-area regret is unchanged (about 0.016 of 0.035). It is still the largest regret source.
- **Why drivable area is stuck:** the label depends only on lane and car-park polygons, the LQR-tracked plan, and
  ego's size. The verifier sees each lane as one averaged vector, no car parks (DP never extracts them), and no ego
  dynamics (the ego token is a constant in every sample).
- **w7 vs w6:** the classifier picks PDM+SL's best candidate most often (top-1 0.848), but its misses cost more,
  so its regret (0.0364) is between w5 and w6. w6 stays the best verifier.
- **Per-metric ceiling (w6):** replacing one metric with its cv label removes this share of w6's regret: drivable
  area 41% (0.0352 → 0.0207), TTC 32%, collision 24%, progress 2%, speed limit 1%, comfort 0%, driving direction
  about 0 (noise). All three multiplicative metrics together: 70%. Progress must be tested with true raw metres; the
  normalized label is zero for unsafe candidates and leaks safety (it alone appears to remove 73%).
- **Scoring is exact:** with true labels for all 7 metrics, our `combined_score` and `first_max` reproduce PDM+SL's
  executed pick in 59,734 of 59,760 samples; the 26 others are near-ties (score gap < 1e-6).
- **Car parks are a minor factor:** they never cause a PDM failure (they count as drivable) and rescue only 4,743 of
  956,160 candidates (0.5%). Only 2,875 of 59,760 samples have mixed drivable labels, and 67 of those involve a
  rescue. The analytic check on DP's 70 input lanes agrees with the label on 98.8% of candidates (4.5% disagreement
  in the mixed samples). The drivable-area problem is lane geometry lost in DP's mean-pooled lane tokens.
- **Multiplicative metrics** (collision, drivable area, driving direction) cause 78% of w6's regret. They are rare,
  geometry-sensitive failures, and the per-metric BCE loss mostly rewards scene-level risk, not which candidate fails.
- **Input caps are fine:** the 5 static-object and 32 agent slots are often full, but dropped objects explain only
  0.37% of samples' collision/TTC failures (`dps-timestep-verifier/runs/slot_check/tests12.csv` on aegis).
- **PDM vs CLS drivable area:** PDM's rule (our label) is stricter than CLS's: lanes only plus car parks, zero
  tolerance, versus CLS's whole drivable layer and 0.3 m tolerance. Part of drivable-area regret may not cost CLS score.

## Next steps

1. Commit the w7 classifier code (as a recorded negative result unless the classifier-as-auxiliary-loss run keeps it).
2. Decide between loss-side and input-side fixes for within-scene safety ranking:
   - loss: a within-scene pairwise loss on the multiplicative metrics (one plan-arch seed, about 23 min);
   - inputs: per-step corner-to-lane distances, raw lane-boundary points or car parks as memory tokens, real ego
     dynamics (needs inputs re-extracted from the simulation logs). Car parks are low priority (see above).
3. Drivable area first: signed per-step corner-to-lane distances plus dense per-step labels; see the
   "Learned Verifier: Future Directions" doc for the full list and order.
4. Planned ablations (also listed in the "Learned Verifier Architecture" doc): drop the ego token; without self-attention
   (cross-attention to the scene only); mark route lanes on the lane tokens and drop the route token (low priority).
5. Closed-loop check of the best verifier on test14-hard (`SELECTOR=verifier`, set `VERIFIER_CKPT`; the default is
   still the old oracle-label `w3_oracle` checkpoint).

## Other artifacts

- Architecture doc: https://claude.ai/artifact/XBsF7EvZaJx1p23ADsqebW
- Future directions (priorities, ceiling results, drivable-area ideas): https://claude.ai/artifact/TcCVpvvn887ZQqY534HVUt
- Per-step verifier attention explainer: https://claude.ai/artifact/ToVJH8MDDVzhfuqMzUyfjH
- Closed-loop videos of the w5 verifier (6 test14-hard scenarios): `videos/verifier_w5cv_seed1_test14-hard_nr_2026-11-19/`
- Remote work runs only on aegis, which is shared with Shubham: check `ps -u shubham` and `nvidia-smi` before
  launching anything.
