# Learned Verifier: Future Directions

Nov 20, 2026 · @Kaushal · Markdown snapshot of the Claude Doc https://claude.ai/artifact/TcCVpvvn887ZQqY534HVUt

**New to the project? Start with [Learned Verifier: Start Here](https://claude.ai/artifact/Y4krwNhPeXfwj64E5QZTZ5).** This doc reflects the state on 2026-11-21. Since then S1 (per-step lane scalars) replaced w6 as the best model at 0.0324 regret, and the dense-label and lane-segment runs finished without beating w6.

Our best verifier is still w6\_timestep (regret 0.0352 ± 0.0013). The next goal is to reduce safety-ranking errors within a scene, mostly drivable area, which is about 0.016 of that regret. The plan: run cheap bounds and readout fixes first, then loss changes, and re-extract inputs only if the bounds say the gain is there.

## Next: three directions from the S1 closed-loop diagnosis

Plan as of 2026-11-27. In closed loop S1 trails PDM+SL by 2.92 CLS, mostly pedestrian collisions: its collision head rates colliding and collision-free candidates about equally safe, so the progress term decides the pick (Doc). The three directions below target that. Two of them need new labels, so one label regeneration run serves all three.

**1. Per-step agent features.** At each of the 41 steps, place every agent with a constant-velocity forecast and take the nearest n to ego's box at that step. Add their features to that step's token, as `--step-lane scalars` adds 14 lane values per step, so self-attention across steps can use them.

- Per agent, in ego's frame at that step: relative position, relative velocity, heading, box size, type (vehicle, pedestrian, cyclist) and the exact box-to-box gap.
- Plus exact summary values per step: the smallest gap to any vehicle and to any pedestrian or cyclist, and a time-to-collision estimate. This is the agent version of S1's lane values, which lifted drivable pair accuracy from 0.57 to 0.66.
- n: start at 8 per step, encoded with a small MLP and max-pooled. With 32 per step, each sample holds 16 × 41 × 32 agent entries. S2 showed that a heavier per-step encoder (36 lane segments) can cost more without beating plain values.
- The label files hold only DP's 32 agents, with pedestrians capped at the nearest 10. Features from them test the idea now but still miss the pedestrian in `159942fd13675580`. The fix is a larger agent pool saved at label time, for example 64 agents with no pedestrian cap.

**2. A stronger collision signal.** Collision and TTC are supervised with one bit per candidate, so the model has to work out which step caused it, and it learns scene-level risk instead.

- Per-step labels: PDM's scorer finds which steps collide and reduces that to one bit. Saving the per-step flags at label time gives a per-step loss on S1's step logits, like the dense drivable labels of the `analytic-pretrain` pilots. Those pilots (A1, A2) test the same idea for drivable area: check their results first.
- No new labels needed: the within-scene pairwise loss on collision (Loss-side fixes) pushes every collision-free candidate's score above every colliding one's in the same scene. That is the ranking that measured 0.25 in `e94b99e48b355de6`.

**3. Progress relative to the human driver (`oracle_ep`).** S1 already predicts each candidate's raw progress in metres (`progress_raw`), so it only has to learn one number per scene: how far the human driver goes in 4 s. Progress is then computed exactly from that estimate, as in `oracle_ep`: min(1, max(ego, 2 m) / max(expert, 2 m)), plus the making-progress multiplier.

- Keep the constant-velocity safety labels, which S1 can match at test time. Logged-future safety labels are worth about 0.24 CLS (`oracle_sl` vs `pdm_sl`).
- The hybrid test (PDM+SL's safety metrics × progress against the median of DP's 16 candidates) bounds this direction with one closed-loop batch, before any training.

**Order**

| Phase | What | New labels | Decided by |
| --- | --- | --- | --- |
| A | Hybrid test (bounds direction 3) | no | closed loop, 3 seeds |
| A | Direction 1 on DP's 32 agents, and the pairwise collision loss | no | collision pair accuracy, then closed loop |
| B | One label regeneration: `expert-progress-labels` merged, plus per-step PDM collision and TTC flags, plus a 64-agent pool without the pedestrian cap | about 4 h planned | — |
| C | Directions 1 (full agent pool), 2 (per-step labels) and 3 (predicted expert progress), each on S1, 3 seeds | — | closed loop |

Each direction is judged by closed-loop CLS on Test14-hard-100 (non-reactive, 3 seeds) against S1's 75.60 and PDM+SL's 78.52, since offline regret has already misled once. Also track collision pair accuracy offline, and rerun the probe on the two pedestrian scenes. A 3-seed training run takes roughly half a day on aegis (S1 and S2 took about 14 h for 6 seed runs on 2026-11-23), so phase C is several days of shared GPU time.

- [ ] Check the `analytic-pretrain` pilot results (A1, A2)
- [ ] Phase A: hybrid selector (`pdm_sl` safety × progress against the candidates' median)
- [ ] Phase A: per-step agent features from DP's 32 agents; pairwise collision loss
- [ ] Phase B: label regeneration with expert progress, per-step collision and TTC flags, and a 64-agent pool
- [ ] Phase C: directions 1 to 3 on S1, closed loop

### Direction 3: what we learned (2026-11-27)

A perfect estimate of the human driver's progress (E) is worth +1.31 CLS over PDM+SL, nearly all of `oracle_ep`'s gain. But the human's logged progress is a hard target to predict, and a conservative estimate may be enough.

**True-E ceiling.** New selector `pdm_sl_ep` (branch `true-e-ceiling`, not pushed): PDM+SL's constant-velocity metrics with progress against the human's logged progress and the making-progress multiplier, i.e. `oracle_ep` without logged agent futures. Test14-hard-100, non-reactive, K = 16, noise 1.5.

| Selector | Seed 0 | Seed 1 | Seed 2 | CLS (mean ± sd) |
| --- | --- | --- | --- | --- |
| `pdm_sl` | 78.96 | 77.96 | 78.64 | 78.52 ± 0.51 |
| `pdm_sl_ep` (true E) | 79.64 | 80.35 | 79.52 | **79.83 ± 0.45** |
| `oracle_ep` (true E and logged agent futures) | 80.07 | 80.52 | 79.83 | 80.14 ± 0.35 |

- The progress definition is worth +1.31 (paired by seed: +0.68, +2.39, +0.88); logged agent futures add only 0.31 more.
- The gain is safety, not distance: time-to-collision 78.7% to 81.3% of scenarios, collision-free 89.5% to 90.2%, comfort 93.3% to 93.7%, route progress unchanged at 93.3%. Capping progress at the human's removes the reward for faster plans.
- Selection takes 56 ms per step, against 44 ms for `pdm_sl` and 227 ms for `oracle_ep`.

**How the progress score is built** (`combined_score_expert`): progress = min(1, max(raw, 2 m) / max(E, 2 m)), 0 if ego goes back more than 2 m, with no safety gating; a making-progress multiplier sets the score to 0 when that ratio is below 0.2. PDM's current definition instead gates raw progress by the safety multipliers and divides by the best of the 16 candidates, which rewards the fastest safe-looking plan.

**Raw progress can be computed, but swapping it into S1 does not help.**

- Projecting the 40th planned pose and the current pose onto a centerline chained from DP's route lanes matches PDM's LQR-tracked raw progress closely: mean error 0.77 m, median 0.31 m, 90th percentile 1.26 m, correlation 0.984, within-scene rank correlation 0.965 (3,000 label samples, 48,000 candidates).
- Swapping it into S1's current normalization, without retraining, raises validation regret from 0.0324 ± 0.0010 to 0.0352 ± 0.0013. PDM's exact raw progress gives 0.0321 ± 0.0022, so raw-progress accuracy is not what limits S1. About two-thirds of the increase is below 3 m/s, where dividing by a small best progress turns about 0.5 m of creep in near-stopped plans into a large error.
- S1's learned raw progress is not a hidden safety signal: within mixed scenes it overstates colliding candidates' progress by an extra 0.33 m and TTC failures' by 0.42 m relative to safe ones, rather than understating them.
- Correction: `train-labels-2k-ego` holds the human's logged speed and acceleration at each sample's time (`patch_ego_state.py` reads the database), not the PDM-driven ego's, so speed bands built from it are the human's speed.

**Offline check of training-free E estimates** (3,000 label samples; true E from the human's logged poses, mean 19.4 m).

| Estimate | Mean error | Mean error on starts and stops | Bias | Correlation |
| --- | --- | --- | --- | --- |
| `ca`: constant acceleration, human's logged speed and acceleration | 2.08 m | 2.98 m | +1.00 m | 0.989 |
| `cv`: constant velocity, human's logged speed | 3.25 m | 7.65 m | +0.07 m | 0.959 |
| `median` of the 16 candidates' progress | 5.35 m | 9.63 m | −1.45 m | 0.891 |
| `first` candidate (noise 1.5 in the labels, not DP's own plan) | 5.44 m | 9.66 m | −1.49 m | 0.888 |
| `cv_plan`: constant velocity, simulated ego's speed from the plans | 7.16 m | 12.62 m | −0.66 m | 0.816 |

Starts and stops are the 942 samples where the human's progress differs from constant velocity by more than 4 m.

- The same constant-velocity rule is off by 3.25 m from the human's speed but 7.16 m from the simulated ego's: most of the gap is the ego not driving like the human, not prediction error.
- Single-step selection cannot measure the benefit. With PDM+SL's exact safety labels, its current definition already matches the true-E pick on 96.6% of samples (regret 0.0013), better than any estimate. The closed-loop gain must build up across steps.

**Why a learned E predictor is hard.**

- The ceiling's E is the human's progress from where the human was. Once the ego drifts from the log, that is not a function of the ego's state, so no predictor can match it exactly. The usable question is how far a human would drive from where the ego is now.
- The hard cases are the ones that cost CLS: starts, stops and yield-or-go moments have two plausible outcomes. A mean lands between them, and a model that leans on current speed predicts going on exactly where a human stops.
- Accuracy may not be needed. The gain needs E not to be too high; an E that is too low only creates ties that safety decides.

**Free baselines in closed loop** (rule-based: PDM+SL's constant-velocity metrics × progress against an E estimate available at test time; no learned model). Test14-hard-100, non-reactive, K = 16, 3 seeds, run 2026-11-27.

| Selector (E estimate) | Noise | Seed 0 | Seed 1 | Seed 2 | CLS (mean ± sd) | vs `pdm_sl` at the same noise, paired |
| --- | --- | --- | --- | --- | --- | --- |
| `pdm_sl` (no E) | 1.5 | 78.96 | 77.96 | 78.64 | 78.52 ± 0.51 | — |
| `pdm_sl_ep_cv`: constant velocity | 1.5 | 79.47 | 79.54 | 78.51 | **79.17 ± 0.58** | **+0.65** |
| `pdm_sl_ep_median`: median of the 16 candidates | 1.5 | 78.98 | 79.19 | 78.03 | 78.73 ± 0.62 | +0.21 |
| `pdm_sl_ep_ca`: constant acceleration | 1.5 | 77.82 | 77.75 | 78.92 | 78.16 ± 0.66 | −0.36 |
| `pdm_sl_ep`: true E (ceiling) | 1.5 | 79.64 | 80.35 | 79.52 | 79.83 ± 0.45 | +1.31 |
| `pdm_sl` (no E) | 0.5 / 1.5 | 78.81 | 77.77 | 78.53 | 78.37 ± 0.54 | — |
| `pdm_sl_ep_first`: DP's own plan | 0.5 / 1.5 | 77.63 | 78.26 | 77.62 | 77.84 ± 0.37 | −0.53 |

- Constant velocity, the simplest estimate, recovers about half of the ceiling's gain (+0.65 of +1.31), within seed noise. Time-to-collision 79.3% of scenarios against 78.7% for `pdm_sl` and 81.3% for the ceiling.
- The offline ranking did not predict closed loop: constant acceleration had the lowest offline error but is the worst here, and DP's own plan scores below `pdm_sl`. A likely cause, not checked, is that estimates which run high (constant acceleration's +1.0 m offline bias) bring back the reward for speed.
- None of these involve S1. The next test combines S1's safety predictions with constant-velocity E (progress capped at the estimate, no safety gating), at inference, without retraining.

**S1 with constant-velocity E at inference** (selector `verifier_ep_cv`, run 2026-11-28; S1's own safety predictions and learned raw progress, progress scored as min(1, max(raw, 2 m) / max(E, 2 m)) with no safety gating and the making-progress multiplier, E = ego speed × 4 s; no retraining). Same settings, 3 seeds.

| Selector | Seed 0 | Seed 1 | Seed 2 | CLS (mean ± sd) | Collision-free | TTC | Route progress | Comfort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 as is | 77.23 | 74.56 | 75.00 | 75.60 ± 1.43 | 87.0% | 77.7% | 93.1% | 87.7% |
| S1 + constant-velocity E | 76.07 | 75.90 | 74.62 | 75.53 ± 0.79 | 87.5% | 76.7% | 87.5% | 93.7% |

- No gain: comfort rises 6 points to PDM+SL's level, collisions are unchanged and route progress drops 5.6 points (the rule-based version with the same E lost 1 point). Without the speed pull, the pick falls to S1's collision predictions, which still cannot separate candidates.
- Offline the same change cuts regret against a CLS-style target by 29% (0.0313 to 0.0223, 5,970 validation samples); offline single-step regret has now disagreed with closed loop twice.
- Follow-up queued: S1 retrained on that target (S1-EP), with progress computed from the plan in training and closed loop. Design, evidence and run plan: Doc.

**Next**

- [x] Done (see the table above). Free baselines in closed loop: `pdm_sl_ep_median`, `_ca`, `_cv` at noise 1.5, and `_first` at noise 0.5 / 1.5 (candidate 0 = DP's own plan, a learned human-progress predictor conditioned on the ego's actual state), 3 seeds each.
- [ ] If they fall short: conservative estimates, such as the lower of the candidate median and speed limit × 4 s, or a low quantile of the candidates' progress.
- [ ] Only then a learned predictor: condition on the ego's state and the scene, predict a low quantile, and judge it in closed loop only.

## Priorities

All runs use cv labels, `--split-seed 0`, 30 epochs and 3 seeds unless noted. A timestep-arch seed takes about 42 min on aegis (84 s per epoch in w7). Baseline is w6: 0.0352 ± 0.0013.

| # | Experiment | Cost | Why now | Keep if |
| --- | --- | --- | --- | --- |
| 1 | Ceiling: select with the true drivable label (`scripts/eval_analytic_drivable.py`) | no training | bounds how much input-side work on drivable area could gain | done 2026-11-20: true drivable label removes 41% of regret, so #6 is justified |
| 2 | Readout fixes: step-head bias init, driving-direction distance readout, comfort as any-step | small code change, 3 seeds (\~2 h) | three mismatches with PDM found in the label review | regret < 0.0352 − 0.0013 or driving-direction / comfort regret drops |
| 3 | Within-scene pairwise loss on the multiplicative metrics | \~23 min for one plan-arch seed, then 3 timestep seeds | targets the BCE-rewards-scene-risk problem directly | drivable-area or collision pair accuracy rises and total regret does not get worse |
| 4 | Classifier loss as an auxiliary term on w6 | small code change, 3 seeds | w7 has the best top-1 (0.848), w6 the best regret | regret below w6's range |
| 5 | Ablations: no self-attention, no ego token, 3 heads of 64 | 3 seeds each, per arch | tests what the current design actually needs | see Architecture and ablations |
| 6 | Input-side: lane-boundary points, car parks, ego dynamics | re-extract inputs from the simulation logs | the only fix for missing geometry | drivable-area regret drops by a clear share of #1's ceiling |
| 7 | Closed loop on test14-hard with the best verifier | one simulation run | the open-loop gains still need to show up in CLS | CLS at or above PDM+SL's selection |

## Sequencing

&#91;embedded content: experiment order · numbers match the Priorities table\]

Experiments 2 to 4 can run in any order on aegis. Experiment 6 waits on the ceiling check, and closed loop runs once on the combined winner.

## w7 result and comparison

w7 finished. It picks PDM+SL's best candidate most often, but when it misses, the pick costs more than w6's misses. Its regret falls between w5 and w6.

| Run | Regret (3 seeds, best epoch) | Top-1 |
| --- | --- | --- |
| w5\_cv (plan, 7 metric heads) | 0.0370 ± 0.0007 | 0.834 |
| w6\_timestep (per-step, 7 metric heads) | **0.0352 ± 0.0013** | 0.819 |
| w7\_timestep\_classify (per-step, one score, `--classify-target best`) | 0.0364 ± 0.0004 | **0.848** |

w7's best epochs were 20, 21 and 22 for seeds 0 to 2. Runs are in `~/IDL_project/dps-timestep-classifier/runs/w7_timestep_classify/` on aegis. About 22% of validation samples are skipped by the classifier loss because all 16 candidates tie.

Still to do:

- [ ] Compare regret per metric and per scenario type from `diagnostic_cv/`. Check whether w7's misses concentrate on the multiplicative metrics.
- [ ] Decide whether to keep the classifier code. It doesn't beat w6, so keep it only if experiment #4 (classifier loss as an auxiliary term) uses it. Otherwise commit it as a recorded negative result.
- [ ] Update the README run table.

## Ceiling checks

Before spending days on new inputs, measure how much regret disappears if one metric were predicted perfectly. Drivable area is the first to check, since it is the largest single regret source (about 0.016 of w6's 0.035).

- [ ] **True drivable label.** `scripts/eval_analytic_drivable.py` already replaces the verifier's drivable-area probability with the true label, and separately with an analytic corner-to-lane check at tolerances of 0, 0.1, 0.2 and 0.3 m. It is hard-coded to `metrics_oracle` and the old `w3_oracle` checkpoint. Add a `--labels cv` option and run it on the w6 checkpoints, since cv labels are our target.
- [x] **Per-metric oracle substitution.** Done with scripts/oracle\_substitution.py (commit d32fc96); results in the table below.
- [ ] **Analytic drivable module.** If the analytic check agrees with the cv labels well at some tolerance, a hybrid becomes an option: geometry computed directly for drivable area, learned predictions for the rest. Check its runtime per scene against the speed budget, since speed is the reason the verifier exists.

**Per-metric oracle substitution, w6\_timestep.** Each row replaces one metric with its cv label and keeps w6's predictions for the other six. It uses the same combined score and selection, 3 seeds, and the 5,970 validation samples. When a multiplicative metric is swapped, progress is re-gated with it. Run on 2026-11-20; results are in `dps-timestep-verifier/runs/w6_timestep/oracle_substitution_cv/` on aegis.

| Metric swapped for its label | Regret | Share of w6 regret removed | Top-1 |
| --- | --- | --- | --- |
| None (w6 as is) | 0.0352 ± 0.0013 | 0% | 0.819 |
| Drivable area | 0.0207 ± 0.0012 | 41% | 0.850 |
| TTC | 0.0239 ± 0.0006 | 32% | 0.875 |
| No at-fault collision | 0.0269 ± 0.0004 | 24% | 0.849 |
| Progress (true metres, gated by the model's safety) | 0.0346 ± 0.0010 | 2% | 0.788 |
| Speed limit | 0.0347 ± 0.0019 | 1% | 0.842 |
| Comfort | 0.0351 ± 0.0012 | 0% | 0.819 |
| Driving direction | 0.0362 ± 0.0015 | −3% (noise) | 0.832 |
| All three multiplicative | 0.0106 ± 0.0003 | 70% | 0.868 |
| All 7 (sanity check) | 0.0000 | 100% | 1.000 |

- Drivable area is the most valuable metric to fix, then TTC and collision. Progress, speed limit, comfort and driving direction are effectively solved for selection.
- The gains are not additive: the three multiplicative metrics remove 65% one at a time and 70% together.
- Swapping in PDM's normalized progress label appears to remove 73%, but that label is zero for unsafe candidates, so it leaks the true safety metrics. The 2% row is the clean measure.
- Sanity checks: with true labels for all 7 metrics, our `combined_score` and `first_max` reproduce PDM+SL's executed pick in 59,734 of 59,760 samples. The 26 exceptions are near-ties, with score gaps below 1e-6.

Decision: the drivable-area ceiling is 41% of w6's regret (0.0145 absolute), above the proposed one-third threshold, so work on drivable-area inputs is justified. The car-park check (Drivable area section) shows that work should target lane geometry, not car parks.

## Loss-side fixes

The multiplicative metrics (collision, drivable area, driving direction) cause 78% of w6's regret. Per-metric BCE grades each candidate on its own, so it mostly learns how risky a scene is, not which of its 16 candidates fails. The listwise ranking term compares candidates, but only on the combined score, where a safety failure is one of many terms.

- [ ] **Within-scene pairwise loss on the multiplicative metrics.** For each scene and metric, take pairs where one candidate passes and another fails. Apply a margin or logistic loss to the difference of their predicted pass logits. Only scenes with mixed labels contribute, so the loss trains ranking within a scene and ignores scene-level risk. Start with one plan-arch seed (about 23 min) to confirm the signal, then run 3 timestep seeds.
  - Measure within-scene pair accuracy per metric (collision 0.78, drivable area 0.57 in w6), plus total regret.
  - Tune the weight so BCE calibration does not degrade. The combined score multiplies probabilities, so miscalibrated passes still cost regret.
- [ ] **Classifier as an auxiliary loss (#4).** Add w7's cross-entropy over the 16 candidates on a separately pooled score, next to w6's metric heads. w7 picks the best candidate most often (top-1 0.848), and w6 has the lowest regret. The aim is both.
- [ ] **Ranking temperature.** The listwise target uses temperature 0.02. Sweep 0.01 and 0.05 on one seed to see whether a sharper or softer target helps near-ties.

## Input-side fixes

PDM's drivable-area label depends on lane, lane-connector and car-park polygons, the LQR-tracked plan, and ego's size. The verifier sees none of these precisely. Each lane is one vector placed at its midpoint, there are no car parks (DP never extracts them), and the ego token is the same constant in every sample. These fixes are the most expensive because they need inputs re-extracted from the simulation logs. Size them with the ceiling check first.

| Fix | What it adds | Cost | Targets |
| --- | --- | --- | --- |
| Per-step corner-to-lane distance | 4 corner distances per step (0 inside a lane polygon), from `analytic_metrics.corner_lane_distance`, as step features | computed from DP's lane inputs, no re-extraction; NumPy on CPU, so precompute or cache it | drivable area |
| Lane-boundary point tokens | raw boundary points as extra memory tokens, so cross-attention can resolve where the edge is | re-extraction; cross-attention cost grows with the token count | drivable area, driving direction |
| Car-park polygons | car-park areas as memory tokens or in the distance feature | re-extraction from the map | drivable area false fails in car parks |
| Real ego dynamics | current speed, acceleration and size in the ego token in place of DP's zeroed fields | re-extraction | TTC, comfort, all footprint checks |
| LQR-tracked states | step features from PDM's tracked rollout, not finite differences of the raw plan | runs PDM's tracker per candidate (cheap next to full scoring) | comfort, TTC, all footprint checks |

Start with the corner-to-lane distance feature, since it reuses the analytic code. If it closes most of the gap to the ceiling, the token-level fixes may not be needed.

## Drivable area: encoding and training ideas

The lane geometry the drivable-area label needs is already in our inputs. DP's frozen encoder throws it away. Each of the 70 lanes has 20 points with centerline position and left- and right-boundary offsets, so `lane_polygons()` rebuilds exact polygons. `LaneFusionEncoder` then averages over the 20 points (`torch.mean(x, dim=1)`) and places the token at the lane's midpoint. Car parks are the only part that is truly missing.

### Diagnostics

- [x] **Car-park check (done; car parks are a minor factor).** `scripts/check_carpark_drivable.py` on branch `drivable-check` replays PDM's drivable test for every label sample. It takes ego's closed-loop state from the label run's simulation logs, tracks candidates with PDM's LQR simulator, and keeps lanes and car parks apart. Per candidate it records:
  - whether a car park rescued it (leaves every lane, but the label passes);
  - whether it fails near a car-park edge (within 1 m);
  - whether it is off DP's 70 input lanes (`dp_off`).

  Results on all of `train-labels-2k` (commit d32fc96, output in `~/IDL_project/dps-drivable-check/runs/carpark_check/` on aegis):

  | Measure | Value |
  | --- | --- |
  | Replay matches the cv drivable label | 956,160 of 956,160 candidates |
  | Candidates failing drivable area | 90,415 (9.5%) |
  | Samples with mixed drivable labels (the only ones that can cost drivable regret) | 2,875 of 59,760 (4.8%) |
  | Candidates rescued by a car park (leave every lane, label passes) | 4,743 (0.5%) |
  | Mixed samples with a car-park rescue | 67 of 2,875 |
  | Failures with a failing corner within 1 m of a car park | 16,444 (18% of failures). `stopping_with_lead` accounts for 8,210 of them: 92% of its 8,904 failures. |

  Car parks never cause a PDM failure, since they count as drivable. They rescue few plans and almost never decide a mixed scene, so extracting them is low priority. The failures at car-park edges are plans slipping between a lane and a car park, which again points to exact geometry.
- [ ] **Analytic noise floor (partly done).** The check against DP's 70 input lanes, on the same LQR-tracked corners, agrees with the cv label on 98.8% of candidates. It flags 9,624 passing candidates as off-road: 4,743 rescued by car parks, and the rest mostly on PDM lanes that DP's lane polygons miss (4,956 candidates have a corner on a PDM lane but outside every DP polygon). It misses 2,183 real failures. In the 2,875 mixed samples it disagrees on 2,071 of 46,000 candidates (4.5%). That is good enough to pretrain on, but fine-tuning on PDM labels remains necessary. Still to measure: the effect of LQR tracking, by repeating the analytic check on the raw plan instead of the tracked one.
- [ ] **Ceiling** with the true label (see Ceiling checks).

### Scene encoding

| Idea | What changes | Cost |
| --- | --- | --- |
| Signed corner distances as step features | each step gets 4 signed distances from ego's corners to the union of lane polygons (negative inside). `corner_lane_distance` exists but returns 0 inside, so it needs a signed version. A continuous margin also ranks passing candidates by how close they came to failing. | low, CPU precompute |
| Lane segment tokens | a small trainable lane encoder alongside DP's, without the mean pool. For example, 4 segments of 5 points per lane gives 280 tokens, each placed at its own center. | medium, about 2.6× the scene tokens |
| Local attention to boundary points | all 1,400 lane points as keys, with each step attending to its k nearest (for example 32), or a distance bias on the logits | medium |
| Drivable-area raster sampled at the corners | an ego-centric signed-distance raster (for example 0.25 m per pixel) through a small CNN, read at each corner of each step with `grid_sample`. Car parks drop in once extracted. | high, new input pipeline |
| Car-park polygons | extracted from nuPlan's map at data generation, as tokens or into the raster | re-extraction; low priority, since car parks rescue 0.5% of candidates and decide 67 of 2,875 mixed samples |
| Corners instead of the rear axle | ego's 4 corner positions per step as features | low |

### Training

- [ ] **Dense per-step labels.** PDM already computes a per-step non-drivable flag (`_ego_areas[..., NON_DRIVABLE_AREA]`) and reduces it to one bit with `any()`. Dumping it gives 41 labels per candidate, so the model is told which step failed instead of guessing from one bit. The cost is a relabel pass. Analytic per-step labels are a cheaper approximation.
- [ ] **Margin regression.** Implemented as the --margin-weight option, off by default. The current target is the overshoot outside the lanes, which is 0 for every on-lane step, so it adds little: being 0.3 m or 1.5 m off fails the same way. Only a signed distance (distance to the boundary from inside) would give graded signal among the majority of candidates that pass (86% in the 4-scenario smoke test). That signed version is not implemented yet.
- [ ] **Counterfactual near-boundary pairs.** Shift or rotate candidates by 0.1 to 1 m and label them analytically. This gives pairs that differ only in drivable area, exactly the within-scene contrast we lack. Using the margin as a soft target limits the effect of analytic label noise.
- [ ] **Pretrain on analytic labels, then fine-tune on PDM's.**
  1. Pretrain: train the per-step drivable-area failure logit on analytic labels: one per step, 41 per candidate, from DP's own lane inputs. The model is told which step went off the lanes instead of guessing from one bit per candidate. The labels are computed on the GPU per batch (the NumPy version takes 50 ms per sample), so they could also cover perturbed plans beyond the 16 recorded candidates.
  2. Fine-tune: switch to PDM's cv drivable-area bit with the existing smooth max and loss. This corrects for where PDM differs from the proxy: LQR-tracked poses, car parks counting as drivable, and lanes outside DP's 70.

  A softer variant keeps the analytic loss as an auxiliary term instead of two stages. Both are running as pilots on the unchanged w6 architecture, with no new lane tokens (see Dense-label pilots below). They test whether dense supervision helps with the current inputs. A null result would be ambiguous, since the mean-pooled lane tokens may be the bottleneck.
- [ ] **Within-scene pairwise loss and a focal loss** for the rare failures (see Loss-side fixes).
- [ ] **Hybrid instead of learned.** Compute drivable area analytically at inference and learn only the other metrics, if it fits the speed budget (`eval_analytic_drivable.py`).

### Dense-label pilots (running)

The first test of dense supervision keeps the w6 architecture unchanged. It adds no lane tokens and no extra heads. Branch `analytic-pretrain`, commit 8cabfbf, worktree `dps-analytic-pretrain` on the laptop and aegis.

**How it works**

- **Labels.** For every one of the 41 scored states and each corner of ego's box, `corner_lane_distance_torch` computes the distance outside DP's 70 input lane polygons. It runs on the GPU per batch and matches the NumPy version to within 1e-3 m. The NumPy version takes 50 ms per sample, too slow for the data loader. A step's target is 1 if any corner is more than 0.2 m off the lanes. The 0.2 m tolerance closes slivers between rebuilt neighbouring lane polygons.
- **What is trained.** Only the per-step drivable failure logit l\_t that the smooth max already reads, with BCE against the step target. The candidate's drivable probability is still σ(−logsumexp\_t l\_t). Calibrated per-step logits make this behave like "any step off": safe steps near −8 give P(fail) ≈ 1.3% after the log 41 sum, and one step near +3 gives P(fail) ≥ 0.95.
- **Stage B.** The normal w6 loss on the cv drivable bit. Its gradient reaches l\_t through the logsumexp, weighted toward the steps the model rates most off-lane. It corrects the 1.2% of candidates where the analytic check and PDM disagree (LQR tracking, car parks, lanes outside the 70, the tolerance).
- **Optional and off:** `--corner-weight` (which corner went off) and `--margin-weight` (overshoot in metres). Either one adds a `geom_head`.

**Runs** (cv labels, split seed 0; `scripts/run_analytic_pilots.sh`, started 2026-11-20 18:44 on aegis; seed 0 first, then seeds 1 and 2):

| Arm | Setup | Output on aegis | Result |
| --- | --- | --- | --- |
| E0 | w6 | `dps-timestep-verifier/runs/w6_timestep` | 0.0352 ± 0.0013, drivable pair acc 0.57 |
| A1 | w6 loss + `--analytic-weight 0.3`, 30 epochs | `dps-analytic-pretrain/runs/a1_analytic_multitask` | running |
| A2 | `--analytic-only` for 10 epochs, then w6 loss for 30 epochs from that checkpoint | `dps-analytic-pretrain/runs/a2_stage_a`, `.../a2_analytic_pretrain` | queued |

Compare at seed 0 first: regret, top-1 and drivable within-scene pair accuracy, which is now logged every epoch. Keep an arm if its 3-seed regret falls below w6's range (< 0.0339) and drivable pair accuracy rises above 0.57. Re-running `oracle_substitution.py` on a winner shows how much of the 41% drivable headroom is left.

### Recommended order

1. Finish the car-park check and the noise-floor and ceiling diagnostics. They need no training.
2. Signed corner distances plus dense per-step labels together. They are cheap and address both root causes: geometry lost in pooling, and credit assignment from one bit.
3. Analytic pretraining and counterfactual pairs, if step 2 helps and the analytic labels agree well with PDM.
4. Segment tokens or the raster, only if steps 2 and 3 plateau well below the ceiling.

## Proposed: local lane attention

Proposal (2026-11-21, not implemented): split each decoder layer's single cross-attention into a local lane stage and a global scene stage. Each step attends to a geometric encoding of the lanes next to it, then to agents, static objects, route and ego. This follows how PDM computes the labels: a step's drivable area, speed limit and oncoming flag depend on the lanes under ego's corners at that step, while collision and TTC depend on agents across the scene.

### Design

| Stage | Queries | Keys and values | Serves |
| --- | --- | --- | --- |
| 1. Self-attention | the 41 step tokens of one candidate | the same 41 steps | temporal context: speed, curvature, contiguous stretches |
| 2. Local lane attention | each step | its k ≈ 8 nearest lanes, one key per (step, lane) built from relative geometry | drivable area, speed limit, driving direction; lane flags for collision and TTC |
| 3. Global scene attention | each step | 32 agents, 5 static objects, route, ego (39 tokens, against w6's 109) | collision, TTC, progress |

**Repeat the stages per layer.** Each of the 2 layers runs 1 → 2 → 3 → FFN, as w6's decoder layer does with its single cross-attention split in two. A single pass would leave steps unable to share what stage 2 found. Driving direction needs that: PDM scores the longest contiguous off-route stretch (under 2 m, 2–6 m, or more). Comfort pooling and progress also read the final step tokens.

**Stage 2 keys, per (step, lane).** Project the step onto the lane and compute, at that point:

- for each of ego's 4 corners: signed distance to the left boundary, the right boundary and the lane's ends;
- heading difference to the lane, and lane width there;
- lane-connector flag, on-route flag and speed limit.

An MLP maps this vector to the key and value. The geometry comes from the raw lane inputs (20 points × 12 features per lane, already in the model inputs), not from DP's mean-pooled lane tokens. Project onto the left and right boundary polylines directly: the three polylines are resampled to 20 points separately, so point-indexed offsets are skewed on curves. The on-route flag is not in the inputs, but can be recovered by matching each lane against the 25 `route_lanes`. Also add the min signed corner distance over lanes as a plain step feature, so the model starts from the near-answer and attention learns the corrections (LQR drift, tolerance, seams between lanes).

**Cost per scene.**

| Part | Size |
| --- | --- |
| Projection (2,624 corners × 70 lanes × 20 points) | about 3.7 M distances; the brute-force polygon check (7.3 M) costs at most 1.6 ms unoptimized |
| Stage 2 attention | 656 step queries × 8 keys |
| Stage 3 attention | 656 × 39, against w6's 656 × 109 |

Latency should land near w6's 3.4 ms, possibly slightly above. Not yet benchmarked.

**Limits it does not fix.** Coverage stays DP's 70 lanes, chosen around ego's current pose. Car parks are missing. Inputs are raw poses, not LQR-tracked states. The analytic check over the same lanes agrees with PDM on 98.8% of candidates, so the realistic target is most of the 0.0145 drivable headroom, not all of it.

**Open question: does stage 3 need DP's lane tokens?** The no-lane-token test running on aegis (branch `no-lane-tokens`, started 2026-11-21) answers it. If collision and TTC hold up without the lane tokens, stage 3 drops them. If not, keep them in stage 3 as global context.

**Test plan** (one change per step, 3 seeds each; judged on drivable-only regret and drivable within-scene pair accuracy on mixed samples, with the other six metrics' pair accuracy and Brier as a guard):

1. Min signed corner distance as a step feature on w6, no new attention: how much exact geometry alone gives.
2. Add stage 2 to w6's layer, lanes kept in stage 3: local lane attention.
3. Drop the lanes from stage 3 (the full design), if the no-lane-token test says that is safe.

&#91;embedded content: proposed decoder layer · 3 attention stages\]

Only stage 2 is new. Its keys come from the raw lane inputs, computed per step, so each step sees the lane boundaries next to its own corners rather than a mean-pooled summary of each lane.

### Lane encoding options for stage 2

Recommended order: analytic signed distances (E) as the floor, then step-frame boundary snippets (B) with E's distances in each key, with the raster patch (D) as the comparison if B's errors cluster at lane seams. Four choices separate the options:

- **Frame.** Geometry in ego's frame at t = 0 (as now), or in the step's own frame (centred on ego at step t, rotated to its heading). In the step frame, "right-front corner 0.2 m past the right edge" looks the same everywhere, so it is learned once.
- **Granularity of a key:** a whole lane, a local piece of a lane, single boundary points, or raster cells.
- **Union.** A corner is legal inside any lane. Either attention learns the min across per-lane keys, or the union is computed up front (signed-distance field or raster), which resolves overlaps and seams between adjacent lanes.
- **Precision against cost.** The label flips at 0 m with no tolerance, so an encoding accurate only to about 0.25 m blurs the cases that matter.

| Option | Key per step | Union | Precision | Cost and risk |
| --- | --- | --- | --- | --- |
| A. Relative geometry per (step, lane) | projection onto the lane, then hand-picked features: corner distances to both boundaries and the ends, heading difference, width, flags | learned | exact, up to the 20-point resampling | cheap; the model never sees what the features leave out (curvature ahead) |
| B. Boundary snippet per (step, lane), step frame | the nearest left- and right-boundary points around the projection (about ±3 points), in the step frame, through a small PointNet or MLP | learned | exact | cheap; the model sees local shape (curvature, narrowing) as well as distance, so B generalizes A |
| C. Nearest boundary points as tokens | per corner, the k nearest points on any lane boundary (densified), with (dx, dy) in the step frame, left/right tag and lane index | learned, over points | exact | no projection or per-lane bookkeeping; needs a kNN over about 2,800 points per corner; shared edges of adjacent lanes appear twice |
| D. Raster patch | signed-distance grid of the union of the 70 lane polygons, built once per scene; a small footprint-aligned patch read per step with `grid_sample` | computed | cell size (0.2 m cells give about ±0.1 m) | union and seams for free; raster once per scene; the error sits at the decision threshold |
| E. Analytic signed distances | per corner: min signed distance over the union, direction to the nearest edge, nearest lane | computed | exact | about 12 numbers per step, no attention; test plan step 1 |
| F. Implicit SDF-style decoder | a trainable per-lane latent; an MLP of (latent, corner position in that lane's frame) gives an inside/outside logit | learned | exact if learned well | learns the geometry end to end; research-heavy and harder to debug |
| G. Lane topology | successor, predecessor and left/right neighbour links | helps seams | — | not in DP's inputs, needs re-extraction; mostly helps driving direction; lowest priority |

A, B, C, E and F keep exact geometry; D trades precision for a free union. Projecting onto the boundary polylines (A, B, C) avoids the point-indexed skew on curves. A and B also carry lane identity (heading, speed limit, on-route), so they help speed limit and driving direction too; D and E only answer how far inside the road a corner is. Every option inherits DP's 70-lane selection and the missing car parks. Try C or F only if B plateaus well short of the 0.0145 headroom.

### Lane properties to encode

A stage 2 key should carry the lane properties PDM's labels read, each measured against ego's corners at that step, not as a per-lane summary. L1 (lane segment tokens with absolute positions, 0.0375 against w6's 0.0352) suggests the step-relative geometry is what matters, not finer tokens.

| Lane property | Metrics | What PDM checks | In DP's inputs? |
| --- | --- | --- | --- |
| Signed distance from each corner to the left and right boundaries and lane ends | drivable area | any corner outside every lane, connector and car-park polygon at any step, zero tolerance | yes, in the raw 20 points; project onto the boundary polylines to avoid the skew on curves |
| Union and overlap with neighbouring lanes | drivable area | a corner is legal inside any lane, so seams between lanes matter | implicit; learned as a min over keys or computed up front |
| Number of lanes the footprint covers | collision, possibly TTC | being in multiple lanes changes at-fault for lateral collisions (to verify) | derivable from the same corner-in-polygon tests |
| On-route flag | driving direction, progress | longest stretch driven off route lanes (under 2 m, 2–6 m, more); progress is measured along the route | no; recover by matching the 70 lanes against the 25 route\_lanes |
| Heading difference, step to lane | driving direction | separates against-traffic from off-route; picks the lane at overlaps | yes, from the centerline direction |
| Speed limit | speed limit | ego speed against the limit of the lane it is in | yes |
| Lane or lane-connector flag | drivable area, collision | connectors are drivable; intersections hold most seams and overlaps | partly; check how DP marks connectors |
| Traffic-light state | collision | red-light connectors appear to be added as obstacles, so entering one counts as a collision (to verify) | yes, one-hot per lane |
| Lane width at the projection | drivable area | not checked; sets a corner's margin | derivable from the boundaries |
| Curvature just ahead | drivable area, comfort | not checked; predicts whether the next steps cut the edge | derivable from a boundary snippet (option B) |

Priority: signed corner distance first (test plan step 1), then on-route flag and heading difference, then speed limit, connector flag and light state in the key, then footprint lane count. The two rows marked "to verify" come from a reading of PDM's scorer and observation code; confirm them in pdm\_scorer before relying on them. Encoding cannot recover car parks or lanes outside DP's 70.

## Readout fixes

Comparing w6's readouts with PDM's label code (see Doc) turned up three mismatches. Each is a small code change, so they can be tested together in one 3-seed run (#2) and separated afterwards if it helps.

- [ ] **Step-head bias init.** With all per-step logits at 0, the smooth max over 41 steps starts at a failure logit of log 41 ≈ 3.71, so P(pass) ≈ 0.024, not the 0.98 the code comment claims. Initialize the four failure-logit biases near −8 so the model starts near the label base rate. Fix the comment either way.
- [ ] **Driving direction as distance, not any-step.** PDM sums the distance travelled in each contiguous stretch off the route lanes and scores the longest one: under 2 m gives 1, 2 to 6 m gives 0.5, and more gives 0. Replace the smooth max with a per-step "off route" probability times the step length, a soft longest-stretch sum, and soft thresholds at 2 m and 6 m.
- [ ] **Comfort as any-step.** PDM fails comfort if any of 6 bounds is broken at any step. w6 reads it with attention pooling. Move it to the per-step head and smooth max like the other any-step metrics. Adding the yaw rate and jerk of each step as features (bounds 0.95 rad/s and 4.13 m/s³) gives the head the quantities PDM checks.
- [ ] **Progress when ego is stopped.** If the best gated progress is 0.1 m or less, PDM sets progress to 1 for every candidate whose safety product is above 0. `_timestep_forward` uses the safety product itself, so a candidate with a 0.5 collision label gets 0.5. Fix: use `(safe > 0).float()`. On the true labels this changes the pick in 1 of 59,760 samples, so it only matters for exactness.

Measure regret per metric for driving direction and comfort, as well as total regret. The changes are small, and total regret alone may not show them.

## Architecture and ablations

The ablations test what the current design needs. The architecture changes are ideas to try once the loss and readout fixes are settled. Full run plans for the first three are in Doc under Planned ablations.

| Experiment | Question it answers | Change | Keep or drop if |
| --- | --- | --- | --- |
| Without self-attention | Is comparing candidates (plan arch) or mixing steps (timestep arch) worth it? | `--no-self-attn`: a diagonal `tgt_mask` | keep self-attention if dropping it raises regret beyond that arch's range |
| Drop the ego token | Is the constant ego token only an attention sink? | `--no-ego-token`, 108 memory tokens | drop the token if regret stays in range |
| Mark route lanes, drop the route token (low priority) | Does a per-lane on-route flag beat one pooled route token? | learned on-route embedding on lane tokens | keep if regret holds or `changing_lane` regret (0.069) improves |
| 3 heads of 64 | Is a head dim of 32 a low-rank bottleneck (32 < 41 steps, 32 < 109 scene tokens)? | `--num-heads 3`, same parameter count | if it helps, try a head size decoupled from d\_model (6 × 64) |
| Cross-candidate layer | Does the per-step model gain from seeing the other 15 candidates? | pool each candidate, then self-attention across the 16 | keep if within-scene pair accuracy rises |
| Geometric attention bias | Does steering each step toward nearby tokens help? | add a learned function of step-to-token distance to the cross-attention logits | keep if drivable-area or collision pair accuracy rises |

The cross-candidate layer and the within-scene pairwise loss (Loss-side fixes) attack the same problem. Run the loss first, since it is cheaper and changes no architecture.

## Closed-loop evaluation

Open-loop regret is a proxy. The project's claim depends on CLS in simulation, at a lower cost per step than PDM scoring. So far only the w5 verifier has been run closed loop (6 test14-hard videos in `videos/verifier_w5cv_seed1_test14-hard_nr_2026-11-19/`).

- [ ] Point `VERIFIER_CKPT` at the best w6 seed and run `SELECTOR=verifier` on test14-hard. The default checkpoint is still the old oracle-label `w3_oracle`, so set it explicitly.
- [ ] Compare CLS with PDM+SL selection, random selection and the first candidate, in the same simulation mode.
- [ ] Record the time per planning step for the verifier and for PDM scoring of 16 candidates.
- [ ] Repeat for whichever checkpoint wins experiments #2 to #6.
- [ ] Check whether drivable-area regret costs CLS. PDM's rule (lanes, lane connectors and car parks, zero tolerance) is stricter than CLS's (the whole drivable layer, 0.3 m tolerance), so some of our drivable-area regret may not matter in closed loop.

## Training target: expert-relative progress (oracle\_ep)

The headroom above PDM+SL comes from how progress is scored, not from knowing where other agents go. An oracle that scores progress against the human driver reaches 80.14 CLS, against PDM+SL's 78.52. The verifier imitates PDM+SL's labels, so it cannot reach this. Retraining on the expert-relative target is the direction where a learned verifier can beat the rule-based scorer.

**How oracle\_ep scores the 16 candidates at each step** (`OracleExpertProgressSelector` and `combined_score_expert`, branch `expert-progress-labels`, commits 18ae976 and 1d175b9, not merged):

1. Track each candidate with PDM's LQR simulator over 4 s (40 poses), as `pdm_sl` does.
2. Place the other agents at their logged futures instead of constant-velocity forecasts. This only works with non-reactive agents.
3. Compute the 7 metrics with CLS weights (progress 5, TTC 5, speed limit 4, comfort 2), as `pdm_sl` does.
4. Replace PDM's progress, which is each candidate's progress divided by the best of the 16, with progress relative to the human driver's logged progress over the same horizon and centerline: min(1, max(ego, 2 m) / max(expert, 2 m)), and 0 if ego goes back more than 2 m. Matching the human scores 1, and driving faster earns nothing. Near the end of a log, the human's progress is scaled up to the full horizon.
5. Add `ego_is_making_progress` as a gate: a progress ratio below 0.2 sets the score to 0, as in CLS.
6. Score = product of the gates × weighted mean of the rest; execute the highest-scoring candidate.

| Selector | Other agents | Progress | CLS (mean ± sd, 3 seeds) |
| --- | --- | --- | --- |
| `pdm_sl` | constant-velocity forecasts | vs best candidate | 78.52 ± 0.51 |
| `oracle_sl` | logged futures | vs best candidate | 78.76 ± 0.57 |
| `oracle_ep` | logged futures | vs human driver | **80.14 ± 0.35** |

Test14-hard-100, non-reactive, K = 16, noise 1.5. Write-up: `results/expert_progress_oracle.md` on the branch.

- Logged futures add 0.24 CLS. The progress definition adds 1.37, positive on every seed (+0.98, +2.41, +0.73), and changes the pick on 41% of steps.
- The gain is not more progress: route progress stays at about 94% for both. It comes from TTC (78–79% to 81–82% of scenarios) and comfort (91–92% to 93–94%). Capping progress at the human's removes the reward for faster, riskier candidates.
- The human's progress is unknown at test time, so `oracle_ep` cannot be deployed. A learned verifier can learn to estimate it from the scene; a rule-based scorer cannot.
- Current verifiers are trained on `pdm_sl`-style labels. In closed loop (2026-11-25) S1 and w6 gained only +0.47 and +0.39 CLS over the first candidate, against +3.39 for `pdm_sl`.

**To do**

- [ ] Merge `expert-progress-labels` into main (2 commits from Nov 11). Its label recorder also saves `progress_expert`.
- [ ] Regenerate `train-labels-2k` with that recorder. The headroom report planned the full label run at about 4 h. Its simulation logs take about 80 GB; aegis has room after the 2026-11-26 cleanup.
- [ ] Retrain S1 on `combined_score_expert`, 3 seeds, then run closed loop on Test14-hard-100. Keep if CLS beats the current S1 (75.60 ± 1.43); the goal is PDM+SL's 78.52 or above.

## Open questions

- What share of the drivable-area ceiling justifies re-extracting inputs? "About a third" in Ceiling checks is a proposed threshold, not a decision yet.
- Should model selection keep using the same validation split as reporting? All runs pick the best epoch on the 5,970-sample validation set they are scored on, which flatters every run a little.
- Is regret the right headline for the report, or should CLS from the closed-loop runs take that place once it exists?
- How much aegis time is available? It is shared with Shubham. The plan as written is about 25 timestep-arch seed runs, roughly 18 hours of GPU time at 42 min each, before closed loop.
