# Learned Verifier: Future Directions

Nov 20, 2026 · @Kaushal

Our best verifier is still w6\_timestep (regret 0.0352 ± 0.0013). The next goal is to reduce safety-ranking errors within a scene, mostly drivable area, which is about 0.016 of that regret. The plan: run cheap bounds and readout fixes first, then loss changes, and re-extract inputs only if the bounds say the gain is there.

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

## Open questions

- What share of the drivable-area ceiling justifies re-extracting inputs? "About a third" in Ceiling checks is a proposed threshold, not a decision yet.
- Should model selection keep using the same validation split as reporting? All runs pick the best epoch on the 5,970-sample validation set they are scored on, which flatters every run a little.
- Is regret the right headline for the report, or should CLS from the closed-loop runs take that place once it exists?
- How much aegis time is available? It is shared with Shubham. The plan as written is about 25 timestep-arch seed runs, roughly 18 hours of GPU time at 42 min each, before closed loop.
