"""Diagnostic selector: drives exactly like VerifierSelector (same pick rule), and at every step saves, for all K
candidates, the verifier's per-metric probabilities and combined score next to PDM+SL's per-metric scores
(constant-velocity forecasts, the verifier's training target) and PDM+SL's pick. Not part of the package: lives in
runs/diag on aegis and is loaded through PYTHONPATH."""
import os
from pathlib import Path

import numpy as np
import torch

from diffusion_planner.scorer.selectors import PDMSpeedLimitSelector, SelectionContext, VerifierSelector


class VerifierProbe(VerifierSelector):
    def __init__(self, ckpt_path: str, args_file: str, device: str = "cuda", out_dir: str = "probe_out"):
        super().__init__(ckpt_path=ckpt_path, args_file=args_file, device=device)
        self._pdm = PDMSpeedLimitSelector()
        self._out = Path(out_dir)

    def initialize(self, initialization) -> None:
        self._pdm.initialize(initialization)

    @torch.no_grad()
    def select(self, candidates: torch.Tensor, context: SelectionContext) -> torch.Tensor:
        from diffusion_planner.scorer.metrics import combined_score, from_pdm

        inputs = {k: v.to(self._device) for k, v in context.inputs.items()}
        plans = candidates[:, :, 0].to(self._device).float()
        probs = torch.sigmoid(self._model(inputs, plans))           # [1, K, 7]
        scores = combined_score(probs)                              # [1, K]
        best = scores.max(dim=1, keepdim=True).values
        pick = (scores >= best - 1e-6).float().argmax(dim=1)        # VerifierSelector's rule

        pdm_pick = int(self._pdm.select(candidates, context)[0])
        sc = self._pdm._scorer
        pdm_metrics = from_pdm(sc.last_metrics(), sc.speed_limit_compliance())  # [K, 7]

        it = context.planner_input.iteration.index
        token = context.scenario.token if context.scenario is not None else "unknown"
        d = self._out / token
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(
            d / f"{it:04d}.npz",
            iteration=it,
            verifier_probs=probs[0].cpu().numpy(),
            verifier_score=scores[0].cpu().numpy(),
            pdm_metrics=pdm_metrics.astype(np.float32),
            pdm_score=np.asarray(self._pdm.last_scores, dtype=np.float32),
            progress_raw=np.asarray(sc.last_metrics()["progress_raw"], dtype=np.float32),
            pick=int(pick[0]),
            pdm_pick=pdm_pick,
            plans=candidates[0, :, 0].cpu().numpy().astype(np.float32),
        )
        return pick.to(candidates.device)
