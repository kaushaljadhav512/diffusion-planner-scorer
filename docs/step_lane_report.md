# Per-step lane features (S1 vs S2)

Branch `step-lane-features`, code commit 623479c. Trained on aegis from 2026-11-23 00:47 to 14:53.

## Question

Does adding local lane information to each step token of the per-step (timestep) verifier, w6, improve
candidate selection? The rest of the model is unchanged, including Diffusion Planner's 70 lane tokens.

- **S1** (`--step-lane scalars`): w6 plus 14 exact lane scalars per trajectory step.
- **S2** (`--step-lane segments`): S1 plus an attention pool over the 36 nearest lane segments per step.

Command (seeds 0, 1, 2; S2 uses `--step-lane segments`):

```
python -m diffusion_planner.scorer.train --data ~/nuplan/labels/train-labels-2k \
  --out runs/s1_step_lane_scalars/seed<s> --labels cv --split-seed 0 --arch timestep \
  --epochs 30 --seed <s> --step-lane scalars
```

Evaluation is on the cv labels (the PDM+SL target), with 5970 validation samples. The best epoch is chosen by
validation regret, which is the same rule used for w6.

## Results

| Run | Seed | Best epoch | Regret | Top-1 | Drivable pair acc | Drivable failure AUC |
|---|---|---|---|---|---|---|
| S1 | 0 | 18 | 0.0328 | 0.806 | 0.652 | 0.887 |
| S1 | 1 | 20 | 0.0313 | 0.820 | 0.670 | 0.896 |
| S1 | 2 | 18 | 0.0331 | 0.810 | 0.647 | 0.892 |
| S2 | 0 | 9  | 0.0327 | 0.792 | 0.653 | 0.879 |
| S2 | 1 | 20 | 0.0317 | 0.816 | 0.668 | 0.891 |
| S2 | 2 | 22 | 0.0337 | 0.812 | 0.656 | 0.905 |

| Model | Regret (mean ± sd, 3 seeds) | Drivable pair acc |
|---|---|---|
| w6 (baseline) | 0.0352 ± 0.0013 | 0.57 |
| **S1** | **0.0324 ± 0.0010** | **0.656** |
| S2 | 0.0327 ± 0.0010 | 0.659 |

For reference, random selection scores 0.0625 regret and the first candidate scores 0.0625.

## Decision

- **Keep S1.** It meets the keep rule (3-seed regret below 0.0339, and drivable pair accuracy rises from 0.57 to
  0.66). The per-step transformer with lane scalars per step is the current best model.
- **Drop S2.** The segment encoder gives no gain over S1 and makes each training step slower (0.34 s against 0.28 s,
  where w6 takes 0.14 s).

## Open items

- Benchmark S1 inference latency. Computing the lane features takes about 0.19 s per batch of 64, and this cost
  matters for the speed argument against PDM+SL.
