# Diagnostic and analysis scripts (2026-11-25 to 2026-11-28)

One-off analyses behind `docs/run_status.md`, `docs/s1_closed_loop_diagnosis.md`, `docs/future_directions.md` and
`docs/s1_ep_architecture.md`. They were run on aegis from `~/IDL_project/dps-closed-loop/runs/diag/` (or with
`PYTHONPATH` at a worktree of this repo) and have hard-coded aegis paths; their outputs are in `results/diag/`.
Batch scripts that launched the closed-loop, open-loop and training runs are in `scripts/batches/`.

| Script | What it does | Output |
| --- | --- | --- |
| `diag_scenarios.py` | Per-scenario closed-loop loss of S1 / w6 vs `pdm_sl` and `first`, charged to its first cause; open-loop miss-rate failures | `diag_scenarios.txt`, `cl_*.csv`, `ol_miss_verifier_scenarios.csv` |
| `diag_divergence.py` | At-fault collision types per selector; first planning step where S1's pick differs from `pdm_sl`'s (seed-0 logs) | `diag_divergence.txt` |
| `diag_collision.py` | For the pedestrian S1 hits: in S1's 32-agent input (DP's 10-pedestrian cap)? does S1's plan hit it within 4 s (logged path and constant-velocity forecast)? | `diag_collision.txt` |
| `probe_selector.py`, `run_sim_probe.sh`, `diag_probe.py` | Rerun S1 on the two pedestrian scenarios, recording S1's per-metric predictions and PDM+SL's per-metric labels for all 16 candidates every step; decompose which metric decided S1's pick | `diag_probe.txt` |
| `check_progress.py` | Progress computed from the 40th planned pose along chained route lanes vs PDM's LQR-tracked raw progress | `check_progress.txt` |
| `eval_computed_progress.py` | S1 validation regret with its learned raw progress replaced by computed / PDM raw progress | `eval_computed_progress.txt` |
| `checks_progress.py` | S1's learned-progress error on unsafe vs safe candidates; regret change by ego speed | `checks_progress.txt` |
| `check_e_estimates.py` | Offline error of training-free estimates of the human's 4 s progress (E) vs the logged value | `check_e_estimates.txt` |
| `cheap_check_cv_target.py` | S1 picks scored against the CLS-style target (constant-velocity E), as is and with E at inference | `cheap_check_cv_target.txt` |

`run_sim_probe.sh` is `scripts/run_sim.sh` with the verifier selector pointed at `probe_selector.VerifierProbe`
(`PYTHONPATH` must include this directory).
