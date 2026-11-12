# Reactive agreement check

In reactive simulation, how often is PDM+SL's pick also the best candidate when candidates are scored against
reactive (IDM) agents that respond to them? Same metric definitions as the labels; only the agent model differs.

Setup: 100 train-split scenarios (`train-agree-100`, `make_train_filter.py --n 100 --seed 1`), reactive challenge,
K = 16, noise 1.5, seed 0, PDM+SL driving, scored every 2 s: 526 steps (`SELECTOR=agree`, see
`diffusion_planner/scorer/agreement.py`). Full output: `agreement_train-agree-100.txt`; per step:
`agreement_train-agree-100.csv`.

| Pick, scored against reactive agents | Among the best | 95% CI (by scenario) | Regret |
|---|---|---|---|
| PDM+SL (constant velocity) | 93.7% | 90.9-96.2% | 0.0083 |
| Logged futures (current training labels) | 83.8% | 78.4-89.2% | 0.0169 |
| First sample | 21.9% | 17.0-27.2% | 0.0522 |
| Random | 21.2% | | 0.0554 |

- PDM+SL's agreement with reactive scoring (93.7%) matches its agreement with logged-future labels (91.3% on these
  states, 92% in the non-reactive label data): reactive labels add little that PDM+SL misses.
- Logged futures are a worse model of reactive traffic than constant velocity (twice PDM+SL's regret), so the
  verifier's current target is weaker than PDM+SL itself in reactive simulation.
- PDM+SL's misses concentrate in `changing_lane` (regret 0.035), `traversing_pickup_dropoff` (0.033) and
  `near_multiple_vehicles` (0.022); most other types are near 0.
- Fork fidelity: on all 526 steps the forked agents' first step matched the simulation's real next step exactly.

Caveat: "best" is PDM's metrics over the 4 s plan with reactive agents, a proxy for the executed scenario's score.
