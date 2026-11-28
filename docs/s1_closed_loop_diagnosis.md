# S1 Closed-Loop Diagnosis

Nov 27, 2026 · @Kaushal · Markdown snapshot of the Claude Doc https://claude.ai/artifact/VGXMAcTbjSDsHi17aP6biy

S1 trails PDM+SL by 2.92 CLS (non-reactive, Test14-hard-100, 3 seeds), and two pedestrian-crossing scenarios carry two-thirds of that gap: S1 picks faster plans and hits a pedestrian in every seed.

## Where S1 loses points

Collisions cost S1 2.45 of its 2.92-point deficit. Each (scenario, seed) pair where the two selectors differ is charged to its first cause: a multiplier metric (at-fault collision, drivable area, driving direction, making progress; failing one zeroes or halves the score) that S1 fails and PDM+SL passes, otherwise the weighted metric with the largest weighted shortfall. The two directions net out.

| Cause | Net CLS points (S1 − PDM+SL) |
| --- | --- |
| At-fault collision (multiplier) | **−2.45** |
| Comfort | −0.63 |
| Drivable area (multiplier) | −0.28 |
| Time-to-collision | +0.44 |
| Progress | +0.03 |
| Speed limit | −0.03 |
| **Total** | **−2.92** |

- **Concentrated.** S1 loses on average in 43 of the 100 scenarios, but the worst 10 carry 90% of the loss. It loses in all 3 seeds in 23 scenarios and wins in all 3 in 14.
- **Two pedestrian scenarios** of type `waiting_for_pedestrian_to_cross` (`e94b99e48b355de6`, `159942fd13675580`) score 0 with S1 in every seed, against about 96 with PDM+SL. Together they cost 1.93 CLS, two-thirds of the gap.
- **S1 picks worse than doing nothing.** In `e94b99e48b355de6`, DP's first candidate scores 94.
- **w6 fails in the same places** (0, 0, 65 and 0, 0, 0 in those two), so this is systematic, not one checkpoint.

| At-fault collision runs, of 300 | Pedestrians and cyclists | Vehicles | Objects |
| --- | --- | --- | --- |
| `first` | 7 | 28 | 9 |
| PDM+SL | 5 | 23 | 6 |
| S1 | **11** | 24 | 7 |
| w6 | 9 | 25 | 8 |

The extra collisions are with pedestrians and cyclists; vehicle and object collisions are about the same across selectors. By scenario type, S1 loses most in pedestrian crossings (−27.4 per scenario, mean over seeds), left turns (−8.6) and traffic-light intersections (−8.5). It beats PDM+SL in pickup and drop-off (+9.8) and near multiple vehicles (+4.4).

## What happened at the failure

In the pedestrian scenes S1 drives faster than PDM+SL and hits a pedestrian it could see, inside PDM's 4 s horizon. Each scenario starts with both selectors facing identical candidates, so the first different pick is S1's own decision (seed-0 logs).

| Scenario | First different pick | S1's plan vs PDM+SL's (8 s) | Outcome |
| --- | --- | --- | --- |
| `e94b99e48b355de6`, pedestrian crossing | step 0 | 31.6 m vs 19.4 m | hits a pedestrian at 5.8 s at 2.6 m/s; PDM+SL is at 1.0 m/s then |
| `159942fd13675580`, pedestrian crossing | step 0 | 23.0 m vs 25.4 m; S1 then speeds up | hits a pedestrian at 5.1 s at 1.6 m/s; PDM+SL is at 0.4 m/s |
| `2754fbd9c2445dce`, lane change | step 0 | 60.8 m vs 56.4 m; S1 reaches 8.2 m/s vs 5.6 | leaves the drivable area |
| `40e1f34c92255a0e`, intersection | step 3 | S1 slows to 1.3 m/s later, PDM+SL does not | collision with a vehicle or object; cause not checked |

### Why S1 hit the pedestrians

For the pedestrian S1 hits, each planning step before contact is checked three ways. Visibility uses Diffusion Planner's agent rule: the 32 nearest agents, with pedestrians and cyclists capped at the nearest 10 when more than 32 agents are present (`max_ped_bike = 10` in `DataProcessor`). The hit tests use S1's chosen plan as planned, not LQR-tracked, against the pedestrian's logged path (exact, since agents replay the log) and against a constant-velocity forecast, the one PDM's scorer uses.

| Check | `e94b99e48b355de6` (56 pedestrians) | `159942fd13675580` (137 pedestrians) |
| --- | --- | --- |
| Steps observed before contact | 59 | 52 |
| Pedestrian in S1's input | 59 (nearest 1–6) | **25**: out from 0.6 s to 2.6 s, ranked 11th–13th |
| S1's plan hits it within 4 s (logged path) | 50 | 38 |
| Constant-velocity forecast shows the hit within 4 s | 44 | 38 |
| First step with a hit within 4 s | 0 (5.8 s before contact) | 3 (4.8 s before contact) |

- **`e94b99e48b355de6` is a scoring miss.** The pedestrian is in S1's input on every step, the conflict is within 4 s, and even a constant-velocity forecast shows it. At step 0 both selectors had the same 16 candidates, and PDM+SL picked a slower one.
- **`159942fd13675580` is partly an input miss.** The 10-pedestrian cap drops it from 0.6 s to 2.6 s, while S1 accelerates from 0.1 to 1.5 m/s. It is back in view at 3.1 s; by 4.6 s the chosen plan hits it within 0.2 s.
- **The 4 s horizon is not the cause.** The conflict sits inside PDM's 40-step window on most steps; only the first few steps of `159942fd13675580` put it beyond 4 s.

### Why S1 missed them

S1 failed to predict the collisions even with the pedestrian in its input: it rated the colliding plans about as safe as the alternatives, so the progress term decided the pick.

- **The pedestrian was in sight.** It was among S1's 32 input agents on all 59 steps before contact in `e94b99e48b355de6`, and on 25 of 52 in `159942fd13675580`, where the 10-pedestrian cap dropped it from 0.6 s to 2.6 s.
- **The spacing between steps is not the cause.** S1 and PDM+SL's labels both use 41 discrete states at 0.1 s over 4 s. At these speeds ego (under 2.7 m/s) moves under 0.3 m and the pedestrian (about 1.4 m/s) under 0.15 m between states, small next to their boxes, so a collision falling between two states is unlikely (not measured). PDM+SL's labels, on the same discrete states, flag S1's pick as colliding on 52 of 59 and 47 of 52 steps.
- **S1 does no geometric check.** It has no ego or pedestrian boxes; it infers a collision from attention between its step tokens and Diffusion Planner's agent tokens.

A rerun of both scenarios with a probe selector (S1, seed 0; it reproduces the original run exactly, 0.000 m ego gap over 149 steps) records S1's per-metric predictions and PDM+SL's per-metric labels for all 16 candidates at every step.

| Up to contact | `e94b99e48b355de6` | `159942fd13675580` |
| --- | --- | --- |
| Steps | 59 | 52 |
| S1's pick fails PDM+SL's collision check | 52 | 47 |
| …and a collision-free candidate existed | 31 | 10 |
| S1's mean P(no collision), candidates PDM+SL marks colliding / collision-free | 0.923 / 0.936 | 0.912 / 0.908 |
| Share of (colliding, collision-free) pairs where S1 rates the collision-free one safer | **0.25** | 0.997, by about 0.01 |

- **Progress decided the pick.** Averaged over the failing steps where S1 and PDM+SL pick differently, the progress term shifts the log of S1's combined score toward its own pick by +0.093 and +0.041, while the collision term shifts it by +0.002 and −0.002. PDM+SL's labels reject the same picks through collision. At step 0 of `e94b99e48b355de6`, S1's pick makes 9.3 m of progress against 4.2 m for PDM+SL's, S1 rates them 0.96 and 0.97 collision-free, and wins 0.655 to 0.598 on progress alone.
- **In `159942fd13675580` the decisive steps are the first second or so:** after about step 11 PDM+SL itself finds no collision-free candidate. S1 ranks the few early pairs correctly but by about 0.01, which progress outweighs, and the pedestrian leaves its input from step 6.
- **The collision head predicts how risky the scene is, not which candidate collides.** This is the within-scene ranking problem named in Learned Verifier: Future Directions; the within-scene pairwise loss and per-step agent scalars target it directly.

## Open-loop miss-rate failures

The open-loop drop is a separate, smaller problem: wrong route choices at turns, mostly in scenarios where S1 does not lose closed-loop points. The verifiers' OLS is about 3.5 points below PDM+SL's, mostly from the miss-rate gate.

| Miss-rate gate failures, of 100 scenarios | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| `first` | 20 | 18 | 22 |
| PDM+SL | 20 | 18 | 19 |
| S1 | 25 | 22 | 24 |
| w6 | 22 | 26 | 23 |

- Only 10 (scenario, seed) pairs fail with S1 while PDM+SL and `first` pass, in 5 distinct scenarios. Half of those pairs are `starting_left_turn`, then 2 `changing_lane`, 2 `starting_right_turn` and 1 `traversing_pickup_dropoff`. w6 fails 8 of the same 10.
- `0cd6cff135c25fa9`, a left turn, fails with both verifiers in all 3 seeds and never with PDM+SL or `first`. In closed loop it is also S1's drivable-area failure in seeds 1 and 2.
- Overlap with closed loop is small: of the 13 scenarios where S1 loses more than 5 points in closed loop, 2 also have extra verifier miss failures.

## Hypotheses and next steps

The pedestrian checks point to a scoring weakness first and the 10-pedestrian cap second; the speed-bias hypothesis is not tested yet.

- **No per-step agent geometry.** S1's step tokens carry 14 exact lane values per step but nothing exact about agents: it judges a conflict only through cross-attention to Diffusion Planner's pooled agent tokens. In `e94b99e48b355de6` the pedestrian is visible and the conflict is within 4 s on 50 of 59 steps, yet S1 keeps picking that plan. Lane scalars lifted drivable pair accuracy from 0.57 to 0.66; agent scalars are the same fix for collisions.
- **Speed bias from the progress label (untested).** The training label scores progress against the fastest of the 16 candidates, which rewards speed. PDM+SL uses the same term, but its collision check is exact, so speed rarely wins. With a fuzzier collision estimate, S1's picks tip toward speed. The expert-relative progress target (`oracle_ep`) removes this reward.
- **The 10-pedestrian cap (confirmed in one scene).** In `159942fd13675580` the pedestrian S1 hits is outside its input on 27 of 52 steps, while S1 accelerates. PDM+SL checks every agent.

**Next steps**

- [ ] Per-step agent scalars: for each step, the gap from ego's footprint to the nearest vehicle and the nearest pedestrian or cyclist under a constant-velocity forecast, from the full observation, not only DP's 32 agents. Same pattern as S1's `--step-lane scalars`; check first how many training samples have pedestrian collision labels.
- [ ] Agent slots: choose agents by distance to the candidate paths instead of to ego, or raise the 10-pedestrian cap for the verifier's own agent input.
- [ ] Expert-relative progress target: regenerate labels from `expert-progress-labels` and retrain S1 (see Learned Verifier: Future Directions).
- [ ] Hybrid test: PDM+SL's safety metrics × progress against an estimate of the human's progress (median progress of DP's 16 candidates). One closed-loop batch.
- [ ] Comfort as any-step: the readout fix listed in Learned Verifier: Future Directions targets the second-largest cause (−0.63 CLS).
- [ ] Left-turn route choice (`0cd6cff135c25fa9`) and the intersection collision (`40e1f34c92255a0e`): not yet checked.

## Data and scripts

- Closed loop: Test14-hard-100, non-reactive, K = 16, noise 1.5, seeds 0–2, run 2026-11-25 (`first`, `pdm_sl`, S1, w6). Open loop: same settings, run 2026-11-26.
- Checkpoints: S1 `dps-step-lane-features/runs/s1_step_lane_scalars/seed<s>/best.pt`, w6 `dps-timestep-verifier/runs/w6_timestep/seed<s>/best.pt`; the verifier seed equals the sampling seed.
- Seed-0 simulation logs kept for S1, w6 and `pdm_sl`; the others' logs were deleted in the 2026-11-26 cleanup, scores kept.
- Scripts and outputs on aegis in `~/IDL_project/dps-closed-loop/runs/diag/` (copied into the repo on 2026-11-28: scripts in scripts/diag/, outputs in results/diag/): `diag_scenarios.py` (per-scenario causes, open-loop misses), `diag_divergence.py` (collision types, first different pick), `diag_collision.py` (pedestrian visibility and hit timing), each with a `.txt` output, plus CSVs of per-scenario scores. The probe rerun: \`probe\_selector.py\` (the selector), \`run\_sim\_probe.sh\` (run\_sim.sh pointed at it), \`probe\_out/\<scenario>/\<step>.npz\` (per-step records), \`diag\_probe.py\` and \`diag\_probe.txt\` (analysis).
