from typing import Dict, Sequence, Union

import torch

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler


def repeat_batch(tensors: Dict[str, torch.Tensor], K: int) -> Dict[str, torch.Tensor]:
    """
    Repeat every tensor K times along the batch dim, sample-major: row b * K + k is copy k of scene b.

    Args:
        tensors: Dict of [B, ...] tensors (model inputs or encoder outputs)
        K: number of copies per scene

    Returns:
        Dict of [B * K, ...] tensors
    """
    return {
        k: torch.repeat_interleave(v, K, dim=0) if isinstance(v, torch.Tensor) else v
        for k, v in tensors.items()
    }


def sample_candidates(
        model: Diffusion_Planner,
        inputs: Dict[str, torch.Tensor],
        K: int,
        noise_scale: Union[float, Sequence[float]] = 0.5,
        steps: int = 10,
    ) -> torch.Tensor:
    """
    Draw K trajectory candidates per scene with one encoder pass and one batched DPM-Solver run.

    Mirrors the inference branch of Decoder.forward (diffusion_planner/model/module/decoder.py),
    with the initial-noise std and the number of solver steps exposed. With K=1, noise_scale=0.5,
    steps=10 and the same seed it must match model(inputs)[1]["prediction"] (task 1.3).

    Args:
        model: Diffusion_Planner in eval mode
        inputs: normalized model inputs (after observation_normalizer), each [B, ...]
        K: candidates per scene
        noise_scale: std of the initial noise x_T on the future states (upstream: 0.5); a sequence of K values
                     gives each candidate its own std (e.g. candidate 0 at 0.5 = vanilla DP, the rest at 1.5)
        steps: DPM-Solver steps (upstream: 10)

    Returns:
        candidates: unnormalized future states (x, y, cos, sin) of ego + predicted neighbors,
                    [B, K, 1 + predicted_neighbor_num, future_len, 4], i.e. [B, K, 11, 80, 4]
    """
    assert not model.training, "sample_candidates needs the model in eval mode"
    decoder = model.decoder.decoder
    P = 1 + decoder._predicted_neighbor_num
    T = decoder._future_len

    with torch.no_grad():
        # Encode the scene once
        encoding = model.encoder(inputs)['encoding'] # [B, N, D]

        # Extract ego & neighbor current states
        ego_current = inputs['ego_current_state'][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][:, :P - 1, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        # Copy so the caller's dict is not modified; the guidance reads the mask from inputs
        inputs = {**inputs, "neighbor_current_mask": neighbor_current_mask}

        current_states = torch.cat([ego_current, neighbors_current], dim=1) # [B, P, 4]
        B = current_states.shape[0]

        # Expand everything the decoder sees to B * K
        inputs = repeat_batch(inputs, K)
        encoding = torch.repeat_interleave(encoding, K, dim=0)          # [B * K, N, D]
        current_states = torch.repeat_interleave(current_states, K, dim=0) # [B * K, P, 4]
        route_lanes = inputs['route_lanes']
        neighbor_current_mask = inputs["neighbor_current_mask"]

    scales = torch.as_tensor(noise_scale, dtype=torch.float32).reshape(-1)
    assert len(scales) in (1, K), f"noise_scale needs 1 or K={K} values, got {len(scales)}"

    # Noise drawn on CPU like upstream, candidate-major: the first B draws are candidate 0 of every scene, the same
    # values a K=1 call (vanilla DP) draws at the same seed. For B=1 this is the same stream as a [B * K] draw.
    noise = torch.randn(K, B, P, T, 4) * scales.expand(K)[:, None, None, None, None]
    noise = noise.transpose(0, 1).reshape(B * K, P, T, 4).to(current_states.device) # row b * K + k
    xT = torch.cat([current_states[:, :, None], noise], dim=2).reshape(B * K, P, -1) # [B * K, P, (1 + V_future) * 4]

    def initial_state_constraint(xt, t, step):
        xt = xt.reshape(B * K, P, -1, 4)
        xt[:, :, 0, :] = current_states
        return xt.reshape(B * K, P, -1)

    model_condition = {
        "cross_c": encoding,
        "route_lanes": route_lanes,
        "neighbor_current_mask": neighbor_current_mask
    }

    # Collision guidance averages its reward over the whole batch, so with guidance enabled
    # the K samples are not fully independent.
    x0 = dpm_sampler(
                decoder.dit,
                xT,
                other_model_params=model_condition,
                diffusion_steps=steps,
                dpm_solver_params={
                    "correcting_xt_fn": initial_state_constraint,
                },
                model_wrapper_params={
                    "classifier_fn": decoder._guidance_fn,
                    "classifier_kwargs": {
                        "model": decoder.dit,
                        "model_condition": model_condition,
                        "inputs": inputs,
                        "observation_normalizer": decoder._observation_normalizer,
                        "state_normalizer": decoder._state_normalizer
                    },
                    "guidance_scale": 0.5,
                    "guidance_type": "classifier" if decoder._guidance_fn is not None else "uncond"
                },
        )
    x0 = decoder._state_normalizer.inverse(x0.reshape(B * K, P, -1, 4))[:, :, 1:] # [B * K, P, V_future, 4]

    return x0.reshape(B, K, P, T, 4)
