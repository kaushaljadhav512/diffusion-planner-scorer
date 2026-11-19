# Verifier training results (week 3)

Per-run `log.jsonl` (metrics per epoch), `best_metrics.json` and `train.log`. Only the `w3_oracle` and `w3_ego` best-epoch checkpoints (`best.pt`) are committed; summary and loading notes in `docs/verifier_heldout_report.md`.

- `overfit`, `overfit_long`: task 2.12 with the old defaults (batch 64, lr 3e-4; 50 and 600 epochs).
- `sweep/`: 100-sample overfit sweep, 50 epochs; name = `lr<lr>_bs<batch>_rw<rank weight>[_t<rank temperature>]`.
  Best: `lr1e-3_bs16_rw3_t0.02` (regret 0.0004, top-1 0.80).
- `w3_oracle`: task 3.1, full data, oracle labels, constant ego state (best regret 0.0362 at epoch 10).
- `w3_ego`: same with the true ego state (`scripts/patch_ego_state.py`; best regret 0.0333 at epoch 19).
- `scripts/ego_speed_probe.py`: how well the candidates imply the true ego speed (R^2 0.84, n=600).

Reference on val: PDM regret 0.0114 / top-1 0.931, First 0.0564 / 0.271, Random 0.0566.
- `w4_cv_paired`: full data, constant-velocity labels (`--labels cv`, the PDM+SL target), constant ego state
  (`train-labels-2k`), 30 epochs; best regret 0.0379 / top-1 0.837 against CV labels at epoch 11 (PDM+SL: 0 / 1.0).
  Other training arguments were not recorded. `diagnostic_cv/`: `diagnose_verifier.py --labels cv`. Drivable area
  (45%) and collision (32%) cause most of the regret, with within-scene pair accuracy 0.57 and 0.53 (near chance).
  Checkpoint not committed (`runs/w4_cv_paired/best.pt` on the GPU machine).
