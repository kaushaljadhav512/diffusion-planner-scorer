# Verifier held-out results (task 3.1)

How well the learned verifier picks among K = 16 candidates on held-out data, compared with PDM+SL and random
selection. Offline only: the verifier has not been run in closed-loop simulation. Data: `results/verifier/w3_oracle`
and `results/verifier/w3_ego` (`best_metrics.json`, `log.jsonl`, `train.log`).

## 1. Setup

| | |
|---|---|
| Labels | `train-labels-2k`: 59,760 samples from 1,992 nuPlan train-split scenarios (see `docs/label_report.md`) |
| Split | by scenario, 10% held out: 53,790 train / **5,970 val** samples (1,793 / 199 scenarios) |
| Target | combined score of the oracle labels (PDM scoring with the logged agent futures, CLS weights) |
| Model | 1.46 M trainable parameters over DP's frozen scene and route encoders |
| Runs | `w3_oracle`: ego state as the planner passes it (velocity, acceleration, steering all 0); best epoch 10 |
| | `w3_ego`: true ego state patched into the inputs (`scripts/patch_ego_state.py`); best epoch 19 |
| Seeds | one seed per run; val regret moves by about ±0.002 between epochs |

Metrics, on the val samples:

- **regret**: combined score of the best candidate minus that of the pick (0 to 1 scale, lower is better).
- **top-1**: fraction of samples where the pick is among the best candidates (ties count).

The PDM baseline ranks candidates by the constant-velocity labels (`metrics_cv`) combined with CLS weights, which
includes the speed-limit term. This is the score `PDMSpeedLimitSelector` uses, so it is the PDM+SL selector.

## 2. Results

| Selector | Regret | Top-1 |
|---|---|---|
| PDM+SL | 0.0114 | 0.931 |
| Verifier, true ego state (`w3_ego`) | 0.0333 | 0.845 |
| **Verifier (`w3_oracle`)** | **0.0362** | **0.828** |
| First sample | 0.0564 | 0.271 |
| Random | 0.0566 | 0.255 |

Same, on the 79.5% of samples where the candidates' scores are not all tied:

| Selector | Regret | Top-1 |
|---|---|---|
| PDM+SL | 0.0144 | 0.914 |
| Verifier, true ego state (`w3_ego`) | 0.0418 | 0.805 |
| **Verifier (`w3_oracle`)** | **0.0455** | **0.784** |
| First sample | 0.0709 | 0.083 |
| Random | 0.0711 | 0.063 |

## 3. Verifier against PDM+SL and Random (`w3_oracle`)

| | vs PDM+SL | vs Random |
|---|---|---|
| Regret difference | +0.025 (worse) | −0.020 (better) |
| Regret ratio | **3.2×** PDM+SL's | **36%** lower than Random's |
| Top-1 difference | −10 points | +57 points |

- Of the regret gap between Random and PDM+SL (0.045), the verifier closes **45%**. It is closer to Random than to
  PDM+SL on regret, but much closer to PDM+SL on top-1: when it misses the best candidate, it misses by more.
- The tie-free subset gives the same ratios (3.2× PDM+SL, 36% below Random), so tied samples do not distort them.
- With the true ego state the ratios become 2.9× PDM+SL and 41% below Random: a small gain, close to the epoch noise.

## 4. What this means

- Swapping PDM+SL for the current verifier would very likely lose closed-loop score. Offline regret is only a proxy:
  labels come from states that PDM+SL drove into, so in closed loop the verifier will meet states it was not trained
  on, which is likely to widen the gap.
- PDM+SL's regret of 0.011 leaves little room to beat it on accuracy. A cheaper approximation of PDM, or a weighted
  PDM + verifier mix, is the more realistic target.
- Not yet tested: whether more data helps (learning curve), constant-velocity labels, a no-scene ablation, and
  multiple seeds.

## 5. Checkpoints

`results/verifier/w3_oracle/best.pt` and `results/verifier/w3_ego/best.pt` (13 MB each) are the best-epoch weights
behind the numbers above. They contain the full verifier, including DP's frozen encoders. Load with:

```python
from diffusion_planner.scorer.model import Verifier
model = Verifier.load("results/verifier/w3_oracle/best.pt", "checkpoints/args.json", device="cuda")
scores = model.score(inputs, candidates)  # [B, K], higher is better; inputs normalized as in the planner
```

`checkpoints/args.json` is DP's config, downloaded by `setup/bootstrap.sh`.

**Use `w3_oracle` for a drop-in closed-loop selector.** The planner passes ego velocity, acceleration and steering
as 0, which is what `w3_oracle` was trained on. `w3_ego` expects the true ego state in `ego_current_state` and will
see inputs unlike its training data unless the selector fills those values in.
