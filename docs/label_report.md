# Label sanity check (task 2.9)

Checks on the verifier training labels from the full run on `train-labels-2k` (task 2.8), before any training.
Data: `results/label_stats.csv` (per scenario type), produced by `python scripts/label_stats.py
$NUPLAN_HOME/labels/train-labels-2k --csv results/label_stats.csv`. All figures use the oracle labels (logged agent
futures, exact in non-reactive simulation) unless marked `cv` (constant-velocity forecasts, what PDM sees).

## 1. The run

| | |
|---|---|
| Filter | `train-labels-2k`: 2000 train-split scenarios from Pittsburgh and Las Vegas, 14 Test14-hard types, at most 3 per log and type; 1992 survive nuPlan's timestamp filter |
| Candidates | K = 16 per step, noise 1.5; the run is driven by `pdm_sl` selection (non-reactive simulation) |
| Samples | **59,760** = 1992 scenarios × 30 recorded steps, in 1992 scenario directories |
| Disk | 4.6 GB |
| Runtime | 2 h 34 min (21:02 to 23:36), 10 workers, RTX 5080 |
| Failures | 0 failed simulations, 0 OOM kills |

Samples per scenario type are 2,160 to 4,620 (72 to 154 scenarios). The four types capped by the pool size are the
smallest: `following_lane_with_lead` 2,160, `behind_long_vehicle` 2,730, `changing_lane` 4,350, and the rest are
about 4,500 to 4,600.

## 2. Per-metric pass rate (mean over all candidates)

| metric | pass rate |
|---|---|
| no_at_fault_collision | 0.978 |
| drivable_area | 0.905 |
| driving_direction | 0.982 |
| progress | 0.801 |
| ttc | 0.946 |
| speed_limit | 0.974 |
| comfort | 0.998 |

- `drivable_area` is the weakest multiplicative metric: about 1 candidate in 10 leaves the drivable area at noise
  1.5. It is the metric a selector most needs to get right, since it zeroes the score.
- `comfort` is almost always 1, so its head carries little signal.
- `speed_limit` is 0.974 on average over the candidates, so the term has room to matter only in the minority of
  samples where the speed limit is violated.

## 3. Do the candidates differ? (selection statistics)

| statistic | value | meaning |
|---|---|---|
| informative | **0.780** | share of samples whose 16 combined scores are not all tied |
| tie rate | 0.220 | 1 − informative: no selector can do anything here |
| best_not_first | 0.710 | share where candidate 0 is not among the best |
| headroom | **0.0512** | mean (best − candidate 0) combined score: the oracle's offline gain over FirstSample |
| random_regret | 0.0507 | mean (best − mean over candidates): the loss of a random pick |
| cv_top1 | 0.926 | the constant-velocity argmax is among the oracle's best in this share of samples |
| cv_regret | **0.0120** | mean (best − score of the PDM-style pick) |
| cv_agreement | 0.923 | oracle and constant-velocity labels pick the same candidate |

What these say:

- **The labels are mostly informative.** 78% of samples, higher than the roughly two thirds expected from the
  Test14-hard evaluation. Risk R6 (uninformative labels) is smaller than feared, and no reweighting is needed to
  start. The 22% tied samples still contribute BCE signal for the individual metrics.
- **Candidate 0 behaves like a random candidate.** Headroom (0.0512) is almost the same as the random regret
  (0.0507). At noise 1.5 the first candidate is no better than any other, so selection has real room.
- **Oracle and constant-velocity labels agree 92% of the time.** Training on either gives nearly the same targets,
  which supports keeping the ablation 3.2 cheap.
- **PDM-style selection already captures most of the headroom offline.** Its regret is 0.0120 against a headroom of
  0.0512, so it recovers about 77% of the oracle's gain. This sets the bar and the ceiling for the verifier:
  - matching PDM means a regret near 0.012;
  - the most the verifier can add over PDM selection offline is about 0.012 of combined score;
  - its advantage is therefore expected to be cost (about 1 ms against 36 ms per step at K=16), not accuracy, as
    the project plan already assumes.

## 4. By scenario type

| type | informative | best_not_first | headroom | cv_top1 | cv_regret |
|---|---|---|---|---|---|
| behind_long_vehicle | 0.578 | 0.523 | 0.042 | 0.935 | 0.011 |
| changing_lane | 0.890 | 0.806 | 0.075 | 0.916 | **0.023** |
| following_lane_with_lead | 0.877 | 0.803 | 0.042 | **0.825** | **0.030** |
| high_lateral_acceleration | 0.781 | 0.696 | 0.056 | 0.957 | 0.007 |
| high_magnitude_speed | 0.865 | 0.796 | 0.035 | 0.963 | 0.008 |
| low_magnitude_speed | 0.834 | 0.759 | 0.053 | 0.903 | 0.012 |
| near_multiple_vehicles | 0.898 | 0.817 | 0.034 | 0.932 | 0.011 |
| starting_left_turn | 0.862 | 0.776 | 0.060 | 0.928 | 0.010 |
| starting_right_turn | 0.776 | 0.714 | 0.061 | 0.942 | 0.009 |
| starting_straight_traffic_light_intersection_traversal | 0.870 | 0.790 | 0.044 | 0.952 | 0.007 |
| stationary_in_traffic | **0.509** | 0.459 | 0.042 | 0.890 | 0.011 |
| stopping_with_lead | **0.512** | 0.468 | 0.024 | 0.932 | 0.005 |
| traversing_pickup_dropoff | 0.861 | 0.792 | 0.073 | 0.894 | **0.022** |
| waiting_for_pedestrian_to_cross | 0.783 | 0.722 | 0.069 | 0.938 | 0.012 |

- **Low-information types:** `stationary_in_traffic` and `stopping_with_lead` are informative in only about half of
  their samples, and `behind_long_vehicle` in 58%. When the ego is stopped or queued, all candidates tend to
  behave alike. These types also have the lowest headroom (0.024 to 0.042).
- **Where PDM-style selection is weakest:** `following_lane_with_lead` (cv_top1 0.825, regret 0.030),
  `changing_lane` (0.023) and `traversing_pickup_dropoff` (0.022). A learned verifier has the most room to beat
  PDM in these types. Two of them (`following_lane_with_lead`, `changing_lane`) are also among the smallest pools.
- **Where headroom is largest:** `changing_lane` (0.075), `traversing_pickup_dropoff` (0.073) and
  `waiting_for_pedestrian_to_cross` (0.069).

## 5. Caveats

- These are offline combined scores in [0, 1] using CLS's weights, not closed-loop CLS points, so 0.05 of headroom
  does not translate one to one into CLS.
- The labels are on-policy for `pdm_sl`: the recorded states come from `pdm_sl`-driven non-reactive simulation. A
  verifier trained on them is later run closed loop on its own selections, which visit a different state
  distribution. This is the usual covariate-shift risk; it is not measured here.
- The ego-state input is a constant in these labels: `ego_current_state` is hard-coded to zero velocity,
  acceleration and steering at inference (`data_processor.py`). The verifier gets no direct ego-speed signal.
  Whether the candidates already imply it is untested.
- Mean pass rates are over candidates, so they say little about which candidate a selector should pick; only the
  within-sample spread matters for ranking.

## 6. Verdict

The labels pass the sanity checks for starting training: 78% of samples are informative, the first candidate
leaves 0.051 of combined score on the table, and the failure metrics have enough positives and negatives
(`drivable_area` fails in about 10% of candidates). The main bar for the verifier is the 0.012 PDM-style regret.
Next: the overfit check (2.12), then training (3.1).
