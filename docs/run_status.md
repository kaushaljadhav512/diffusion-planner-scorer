# Learned verifier: status of runs

Updated 2026-11-28. **Start here.** This page is the current state of the project and points to everything else.

Goal: a learned verifier that picks among Diffusion Planner's (DP's) K = 16 sampled candidates as well as rule-based
PDM+SL selection, at a lower cost per step. Closed-loop score (CLS) in nuPlan simulation decides; offline regret has
disagreed with closed loop twice (see "What we learned").

**Current best learned scorer: S1-EP** (S1 retrained on a CLS-style progress target, `--progress-mode ep_cv`), 76.84
CLS. It beats S1 (75.60) but is still below PDM+SL (78.52) and vanilla DP (77.43).

## Documents

| Document | Live (Claude Docs) | Snapshot in the repo |
| --- | --- | --- |
| Future directions: plan, ceilings, direction-3 results | https://claude.ai/artifact/TcCVpvvn887ZQqY534HVUt | `docs/future_directions.md` |
| S1 closed-loop diagnosis: why S1 hits pedestrians | https://claude.ai/artifact/VGXMAcTbjSDsHi17aP6biy | `docs/s1_closed_loop_diagnosis.md` |
| S1-EP verifier architecture: design, evidence, results | https://claude.ai/artifact/U25nFyoDXQ2XTB9Kfwwukr | `docs/s1_ep_architecture.md` |
| Verifier architecture (Nov 20) | https://claude.ai/artifact/XBsF7EvZaJx1p23ADsqebW | `docs/verifier_architecture.md` |
| Per-step verifier attention explainer | https://claude.ai/artifact/ToVJH8MDDVzhfuqMzUyfjH | `docs/per_step_verifier_attention.md` |

Snapshots do not follow later edits to the live docs; re-export to refresh them. Other reports: `docs/headroom_report.md`
(best-of-K headroom, weeks 1-2), `docs/step_lane_report.md` (S1 vs S2), `results/expert_progress_oracle.md` (`oracle_ep`).

## Results: Test14-hard-100 (100 scenarios), K = 16, 3 seeds

Non-reactive CLS unless noted; open-loop score (OLS) where run. Noise 1.5 unless noted. Tables:
`results/selectors_test14-hard_seeds.csv` (per config, with per-metric means), `results/selectors_test14-hard_runs.csv`
(per run), `results/open_loop_test14-hard-100.csv` (OLS per metric). Differences under about 1.5 CLS between 3-seed
means are within seed noise.

| Selector | What it is | CLS (mean ± sd) | OLS | Deployable |
| --- | --- | --- | --- | --- |
| `oracle_ep` | PDM with logged agent futures, progress vs the human's logged progress | 80.14 ± 0.35 | | no |
| `pdm_sl_ep` | PDM+SL, progress vs the human's logged progress (true-E ceiling) | 79.83 ± 0.45 | | no |
| `pdm_sl_ep_cv` | PDM+SL, progress vs ego speed × 4 s | **79.17 ± 0.58** | | yes, best deployable |
| `oracle_sl` | PDM+SL with logged agent futures | 78.76 ± 0.57 | | no |
| `pdm_sl_ep_median` | PDM+SL, progress vs median of the candidates' progress | 78.73 ± 0.62 | | yes |
| `pdm_sl` | PDM-Closed scoring + speed-limit metric, CLS weights (the verifier's teacher) | 78.52 ± 0.51 | 67.99 ± 0.60 | yes |
| `pdm_sl_ep_ca` | PDM+SL, progress vs constant-acceleration estimate | 78.16 ± 0.66 | | yes |
| `pdm_sl_ep_first` | PDM+SL, progress vs candidate 0 (noise 0.5 / 1.5) | 77.83 ± 0.37 | | yes |
| vanilla DP | candidate 0 at noise 0.5 | 77.43 ± 0.43 | | yes |
| **S1-EP** | S1 retrained, `--progress-mode ep_cv` | **76.84 ± 1.75** | | yes, best learned |
| S1 | per-step verifier + 14 lane values per step | 75.60 ± 1.43 | 64.17 ± 0.88 | yes |
| S1 + E at inference | S1 checkpoints, `verifier_ep_cv`, no retraining | 75.53 ± 0.79 | | yes |
| w6 | per-step verifier | 75.52 ± 2.25 | 64.36 ± 1.70 | yes |
| `first` | candidate 0 at noise 1.5 | 75.13 ± 0.68 | 67.58 ± 1.56 | yes |

- Reactive CLS: `pdm_sl` 74.18 ± 0.26, `first` 73.00 ± 0.52 (verifiers not run reactive).
- Full Test14-hard (272 scenarios): vanilla DP 74.63 NR / 68.63 R (paper 75.99 / 69.22); `pdm_sl` at noise 1.5
  75.90 ± 0.20 NR / 71.95 ± 0.53 R. No verifier has been run on the full split.
- Per-metric (NR, 3-seed means): S1-EP collision-free 87.8%, TTC 77.3%, route progress 89.6%, comfort 94.7%; S1
  87.0 / 77.7 / 93.1 / 87.7; `pdm_sl` 89.5 / 78.7 / 93.3 / 93.3.
- Offline validation regret (cv labels, split seed 0, 5,970 samples; lower is better, random 0.0625): w5 0.0370, w6
  0.0352, w7 classifier 0.0364, S1 0.0324 ± 0.0010, S2 0.0327. S1-EP 0.0196 ± 0.0009 against its own (different)
  target, so not comparable.

## What we learned (2026-11-25 to 11-28)

1. **Offline regret does not predict closed loop.** S1 halves random's regret offline but keeps only about 14% of
   PDM+SL's closed-loop gain; S1 with E at inference cuts regret 29% offline and gains nothing in closed loop.
2. **S1's loss vs PDM+SL is pedestrian collisions** (`docs/s1_closed_loop_diagnosis.md`): 2.45 of 2.92 points; two
   `waiting_for_pedestrian_to_cross` scenarios carry two-thirds. S1's collision head rates colliding and safe
   candidates alike (pair accuracy 0.25 in one scene), so the progress term decides. In one scene the pedestrian was
   outside DP's agent input (the nearest 10 pedestrians only, `max_ped_bike = 10`). The 4 s horizon and the spacing of
   the 41 scored states are not the cause.
3. **The headroom above PDM+SL is in the progress definition, not in agent forecasts.** Progress vs the human's logged
   progress adds +1.31 CLS (ceiling 79.83); logged agent futures add only 0.31. The gain is TTC and comfort; route
   progress is unchanged. Constant-velocity E recovers about half (79.17).
4. **Raw progress can be computed** from the 40th planned pose along chained route lanes (`route_progress.py`): 0.77 m
   mean error vs PDM's LQR-tracked label. Swapping it into S1 without retraining does not help; S1's learned progress is
   not a hidden safety signal.
5. **`train-labels-2k-ego` holds the human's logged speed**, not the PDM-driven ego's. The simulated ego's speed comes
   from the label run's simulation logs (sidecar `~/nuplan/labels/train-labels-2k-epcv`).
6. **A learned E predictor is hard and may not be needed**: the logged human progress is not a function of the ego's
   state once it drifts from the log; E only needs to not be too high.
7. **S1-EP** (retrained on PDM+SL labels with progress vs ego speed × 4 s, progress computed not learned): +1.24 over S1
   and +1.31 over the inference-only change (every seed). The gain is comfort; collisions and TTC are unchanged.

## Next steps

1. Collision ranking, on top of S1-EP: per-step agent features (gap to the nearest vehicle and pedestrian under a
   constant-velocity forecast, from the full observation) and a within-scene pairwise collision loss.
2. Agent input: pick agents by distance to the candidate paths, or lift the 10-pedestrian cap for the verifier.
3. A minimum E for a stopped ego (route progress: S1-EP 89.6% vs PDM+SL 93.3%).
4. S1-EP on the full Test14-hard and with reactive agents; benchmark selection time per step against PDM+SL (44 ms).
5. Unchecked: the left-turn route errors (`0cd6cff135c25fa9`) and the intersection collision (`40e1f34c92255a0e`).

## Where things live (aegis)

All remote work runs on aegis (`ssh aegis`, account shared with Shubham). Ask before using its GPU; check
`nvidia-smi` and `ps -u shubham` first. Its git identity is Shubham's: commit with
`git -c user.name="Kaushal Jadhav" -c user.email=...`. No Claude attribution in commits or PRs.

| What | Path on aegis |
| --- | --- |
| S1-EP checkpoints, logs, closed-loop batch logs | `~/IDL_project/dps-true-e/runs/s1_ep_cv/seed{0,1,2}/` (`best.pt`, `log.jsonl`), `runs/*.log` |
| S1 checkpoints | `~/IDL_project/dps-step-lane-features/runs/s1_step_lane_scalars/seed{0,1,2}/best.pt` |
| w6 checkpoints | `~/IDL_project/dps-timestep-verifier/runs/w6_timestep/seed{0,1,2}/best.pt` |
| Labels | `~/nuplan/labels/train-labels-2k` (59,760 samples), `-ego` (human's speed patched), `-epcv` (S1-EP sidecar) |
| Simulation outputs | `$NUPLAN_EXP_ROOT/exp/simulation/<challenge>/<run>/test14-hard-100/<timestamp>/` (scores kept for every run) |
| Kept simulation logs | seed 0 of S1, w6, `pdm_sl` (NR, Test14-hard-100); the label run `dp_k16_label_noise1.5/train-labels-2k/2026-11-01-21-02-35` (needed for the sidecar and `check_carpark_drivable.py`) |
| Diagnostic scripts and outputs | `~/IDL_project/dps-closed-loop/runs/diag/` (copied to `scripts/diag/`, `results/diag/`) |
| Cleanup record (2026-11-26, about 140 GB freed) | `~/cleanup_2026-11-26.txt` |

Worktrees on aegis: `dps-true-e` (branch `true-e-ceiling`, at `f8251c2`, behind `main` by doc commits only),
`dps-closed-loop` (`closed-loop`), `dps-step-lane-features`, `dps-timestep-verifier`, `dps-analytic-pretrain`,
`dps-drivable-check`, `dps-no-lanes`, `dps-timestep-classifier`, `dps-verifier-ablations`; `diffusion-planner-scorer`
is an old `main` checkout (the conda env's editable install points there, so set `PYTHONPATH` to the worktree).

## Code map

| Piece | Where |
| --- | --- |
| Selectors: `first`, `random`, `pdm`, `pdm_sl`, `pdm_sl_ep`, `pdm_sl_ep_<median\|first\|cv\|ca>`, `oracle*`, `verifier`, `verifier_ep_cv` | `diffusion_planner/scorer/selectors.py`, `SELECTOR=` in `scripts/run_sim.sh` |
| Challenges `nr`, `r`, `ol` (open loop); `VERIFIER_TAG`, `VERIFIER_RAW=predicted\|computed` | `scripts/run_sim.sh` |
| Verifier training, `--progress-mode pdm\|ep_cv`, `--sidecar` | `diffusion_planner/scorer/train.py` |
| Computed route progress, CLS-style progress score | `diffusion_planner/scorer/route_progress.py` |
| S1-EP sidecar builder | `scripts/make_epcv_sidecar.py` |
| Batch scripts used for these runs | `scripts/batches/` |
| Diagnostics (with README) | `scripts/diag/`, outputs in `results/diag/` |
