# Verifier training results (week 3)

Per-run `log.jsonl` (metrics per epoch), `best_metrics.json` and `train.log`. Only the `w3_oracle` and `w3_ego` best-epoch checkpoints (`best.pt`) are committed; summary and loading notes in `docs/verifier_heldout_report.md`.

- `overfit`, `overfit_long`: task 2.12 with the old defaults (batch 64, lr 3e-4; 50 and 600 epochs).
- `sweep/`: 100-sample overfit sweep, 50 epochs; name = `lr<lr>_bs<batch>_rw<rank weight>[_t<rank temperature>]`.
  Best: `lr1e-3_bs16_rw3_t0.02` (regret 0.0004, top-1 0.80).
- `w3_oracle`: task 3.1, full data, oracle labels, constant ego state (best regret 0.0362 at epoch 10).
- `w3_ego`: same with the true ego state (`scripts/patch_ego_state.py`; best regret 0.0333 at epoch 19).
- `scripts/ego_speed_probe.py`: how well the candidates imply the true ego speed (R^2 0.84, n=600).

Reference on val: PDM regret 0.0114 / top-1 0.931, First 0.0564 / 0.271, Random 0.0566.
- `learning_curve/`, `learning_curve.csv`: learning curve (`scripts/run_learning_curve.sh`): trained on 10 / 25 / 50 /
  100% of the training scenarios (3 seeds each, same 20-epoch step budget), evaluated on w3_oracle's validation set.
  Regret 0.0375 ± 0.0006 (179 scenarios), 0.0357 ± 0.0021 (448), 0.0362 ± 0.0008 (896), 0.0354 ± 0.0001 (1793);
  top-1 0.811 to 0.849. Flat from 25% on: more labels of the same kind will not close the gap to PDM+SL (0.0114).
  Checkpoints are not committed.
