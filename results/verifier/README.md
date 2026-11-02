# Verifier training results (week 3)

Per-run  (metrics per epoch),  and . Checkpoints are not committed.

- , : task 2.12 with the old defaults (batch 64, lr 3e-4; 50 and 600 epochs).
- : 100-sample overfit sweep, 50 epochs; name = .
  Best:  (regret 0.0004, top-1 0.80).
- : task 3.1, full data, oracle labels, constant ego state (best regret 0.0362 at epoch 10).
- : same with the true ego state (; best regret 0.0333 at epoch 19).
- : how well the candidates imply the true ego speed (R^2 0.84, n=600).

Reference on val: PDM regret 0.0114 / top-1 0.931, First 0.0564 / 0.271, Random 0.0566.
