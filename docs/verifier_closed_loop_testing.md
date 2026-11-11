# Testing a learned selector in closed loop

Instructions for running a verifier-based selector in nuPlan simulation and comparing it with the existing
selectors. Offline results for the checkpoints are in `docs/verifier_heldout_report.md`.

## Before you start: two things to know

1. **Use `w3_oracle` as the drop-in.** The planner sets ego velocity, acceleration and steering to zero, which is
   what `results/verifier/w3_oracle/best.pt` was trained on. `results/verifier/w3_ego/best.pt` expects the true
   values in `ego_current_state`, so a selector using it would have to supply them (see
   `scripts/patch_ego_state.py` for how the training data was patched), or that model will see inputs unlike its
   training data.
2. **Loading needs the base planner's config:**
   ```python
   Verifier.load("results/verifier/w3_oracle/best.pt", "checkpoints/args.json")
   ```
   `checkpoints/` isn't in git, but `setup/bootstrap.sh` downloads `args.json` (and DP's `model.pth`) into it.

## 1. Setup

```bash
cd ~/diffusion-planner-scorer            # wherever the repo is
conda activate diffusion_planner
source setup/config.env
git pull
ls checkpoints/args.json checkpoints/model.pth   # if missing: bash setup/bootstrap.sh
```

Quick load check (should print about 1.46 M trainable parameters):

```bash
python - <<'EOF'
import torch
from diffusion_planner.scorer.model import Verifier
m = Verifier.load("results/verifier/w3_oracle/best.pt", "checkpoints/args.json", device="cuda")
print(sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6, "M trainable")
EOF
pytest tests/test_verifier.py
```

## 2. What the selector receives

A selector subclasses `AbstractSelector` (`diffusion_planner/scorer/selectors.py`) and implements
`select(candidates, context) -> indices [B]`. In `MultiSampleDiffusionPlanner.compute_planner_trajectory`:

- `candidates` is `[B, K, P, T, 4]`: all P predicted agents, unnormalized. **The verifier wants the ego plans
  only, `candidates[:, :, 0]` (`[B, K, T, 4]`)**. It normalizes them itself.
- `context.inputs` holds the model inputs **already normalized** by the planner, which is what the verifier expects.
- Load the checkpoint once in `__init__`, not per step. Use `torch.no_grad()` in `select`.
- On ties, pick the lowest index (as `PDMSelector` does), so the selector falls back to candidate 0.

Minimal version:

```python
class VerifierSelector(AbstractSelector):
    def __init__(self, ckpt_path: str, args_file: str, device: str = "cuda"):
        from diffusion_planner.scorer.model import Verifier
        self._model = Verifier.load(ckpt_path, args_file, device=device)

    @torch.no_grad()
    def select(self, candidates, context):
        scores = self._model.score(context.inputs, candidates[:, :, 0])   # [B, K]
        return scores.argmax(dim=1)
```

The verifier runs its own copy of DP's encoder here (about 3.4 ms per step). Reusing the planner's encoding
(`encoding=` argument of `score`) would bring it to about 1 ms, but the planner does not expose it yet.

## 3. Register it in the run script

`scripts/run_sim.sh` maps `SELECTOR=<name>` to a class. Add a case next to `pdm_sl`:

```bash
      verifier)
        SELECTOR_TARGET=diffusion_planner.scorer.selectors.VerifierSelector
        SELECTOR_ARGS+=(
          "++planner.diffusion_planner_k.selector.ckpt_path=$DP_ROOT/${VERIFIER_CKPT:-results/verifier/w3_oracle/best.pt}"
          "++planner.diffusion_planner_k.selector.args_file=$DP_ROOT/checkpoints/args.json"
        ) ;;
```

Also add `verifier` to the error message list in the `*)` case.

## 4. Runs

Use **noise 1.5**, the noise the training labels were recorded at and the one the baselines below use.

1. Smoke test (5 scenarios, NR):
   ```bash
   SMOKE=5 NOISE=1.5 SELECTORS="first verifier" CHALLENGES=nr bash scripts/run_headroom.sh
   ```
   **Done when:** both runs show `failed` = 0 and the verifier's CLS is in a plausible range (not near 0).
2. Subset evaluation, 100 Test14-hard scenarios, 3 seeds, NR and R (a 9-run grid took about 65 min on the GPU machine):
   ```bash
   for seed in 0 1 2; do NOISE=1.5 SEED=$seed SELECTORS="first verifier" bash scripts/run_headroom.sh; done
   ```
   `first` is rerun so that `verifier` gets a same-seed `vs_first` column.
3. Commit `results/headroom_test14-hard-100{,_seeds}.csv`, then the per-scenario comparison:
   ```bash
   python scripts/compare_scenarios.py --split test14-hard-100 --config dp_k16_verifier_noise1.5 --challenge nr
   ```

## 5. What to compare against

Existing results on the same subset (K = 16, noise 1.5, 3 seeds, mean ± std):

| Selector | NR | R |
|---|---|---|
| Vanilla DP (First, noise 0.5) | 77.43 ± 0.43 | 70.66 ± 0.50 |
| First | 75.13 ± 0.68 | 73.00 ± 0.52 |
| PDM+SL | 78.52 ± 0.51 | 74.18 ± 0.26 |

Differences under about 1.5 CLS are seed noise. Offline, `w3_oracle` closed 45% of the regret gap between Random
and PDM+SL, so a CLS between First and PDM+SL is the expected result. Report as well:

- the selection runtime per step (`select_ms` column of `scripts/summarize_results.py`, from `selection_runtimes` in the run's
  `runner_report.parquet`) next to `pdm_sl`'s;
- whether the closed-loop result is worse than offline regret suggests. Labels come from states that PDM+SL drove
  into, so a larger gap in closed loop points to distribution shift.
