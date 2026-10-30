"""
Learned verifier for best-of-K Diffusion Planner (task 2.11).

Scene: DP's frozen encoder (agents, static objects, lanes: 107 tokens x 192) plus DP's frozen route encoder
(1 token) and a learned ego-state token. Candidates: one token per ego plan, an MLP over its normalized 80 x 4
states. Transformer decoder layers: candidates attend to each other (self-attention) and to the scene
(cross-attention). Heads: one logit per metric of diffusion_planner.scorer.metrics, trained with BCE against the
PDM labels; the combined score uses CLS's formula.

    model = Verifier.from_dp_checkpoint("checkpoints/args.json", "checkpoints/model.pth")
    scores = model.score(inputs, candidates)   # [B, K], higher is better
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from timm.models.layers import Mlp

from diffusion_planner.model.diffusion_planner import Diffusion_Planner_Encoder
from diffusion_planner.model.module.decoder import RouteEncoder
from diffusion_planner.scorer.metrics import METRICS, combined_score
from diffusion_planner.utils.config import Config


def scene_padding_mask(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    The padding mask DP's encoder uses for its 107 tokens (agents, static objects, lanes): True = empty slot.

    Returns:
        mask: [B, agent_num + static_objects_num + lane_num] bool
    """
    agents = (inputs["neighbor_agents_past"][..., :8] != 0).sum(dim=(-1, -2)) == 0
    static = (inputs["static_objects"][..., :10] != 0).sum(dim=-1) == 0
    lanes = (inputs["lanes"][..., :8] != 0).sum(dim=(-1, -2)) == 0
    mask = torch.cat([agents, static, lanes], dim=1)
    mask[:, 0] = False # as FusionEncoder: never mask everything
    return mask


class Verifier(nn.Module):

    def __init__(self, config, depth: int = 2, num_heads: int = 6, dropout: float = 0.1, use_scene: bool = True):
        """
        Args:
            config: DP config (diffusion_planner.utils.config.Config or the training args)
            depth: transformer decoder layers over the candidates
            use_scene: False drops the encoder and route tokens (trajectory-only ablation, task 3.3); the ego-state
                       token stays
        """
        super().__init__()
        dim = config.hidden_dim
        self.hparams = {"depth": depth, "num_heads": num_heads, "dropout": dropout, "use_scene": use_scene}
        self.use_scene = use_scene

        # Frozen DP modules (weights from the DP checkpoint, see from_dp_checkpoint)
        self.encoder = Diffusion_Planner_Encoder(config)
        self.route_encoder = RouteEncoder(
            config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=dim)
        for module in (self.encoder, self.route_encoder):
            module.requires_grad_(False)

        # DP's normalization of ego future states (x, y, cos, sin)
        self.register_buffer("plan_mean", torch.as_tensor(config.state_normalizer.mean, dtype=torch.float32)[0, 0])
        self.register_buffer("plan_std", torch.as_tensor(config.state_normalizer.std, dtype=torch.float32)[0, 0])

        self.scene_proj = nn.Linear(dim, dim)
        self.ego_proj = Mlp(in_features=10, hidden_features=dim, out_features=dim, act_layer=nn.GELU)
        self.plan_proj = Mlp(in_features=config.future_len * 4, hidden_features=2 * dim, out_features=dim, act_layer=nn.GELU)

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(dim, num_heads, dim_feedforward=4 * dim, dropout=dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, len(METRICS))

    def train(self, mode: bool = True):
        super().train(mode)
        # The frozen DP modules always run as at inference (no dropout / drop-path)
        self.encoder.eval()
        self.route_encoder.eval()
        return self

    def encode(self, inputs: Dict[str, torch.Tensor], encoding: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: normalized model inputs, each [B, ...]
            encoding: DP's encoder output for these inputs if already computed (the sampler's), [B, 107, D]

        Returns:
            memory: [B, N, D] scene tokens; mask: [B, N] bool, True = padding
        """
        ego = self.ego_proj(inputs["ego_current_state"].float())[:, None] # [B, 1, D]
        B = ego.shape[0]
        if not self.use_scene:
            return ego, torch.zeros(B, 1, dtype=torch.bool, device=ego.device)

        with torch.no_grad():
            if encoding is None:
                encoding = self.encoder(inputs)["encoding"]                # [B, 107, D]
            route = self.route_encoder(inputs["route_lanes"])[:, None]     # [B, 1, D]
        memory = torch.cat([self.scene_proj(torch.cat([encoding, route], dim=1)), ego], dim=1)
        mask = torch.cat([scene_padding_mask(inputs), torch.zeros(B, 2, dtype=torch.bool, device=ego.device)], dim=1)
        return memory, mask

    def forward(self, inputs: Dict[str, torch.Tensor], candidates: torch.Tensor,
                encoding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: normalized model inputs, each [B, ...]
            candidates: [B, K, T, 4] ego plans (x, y, cos, sin), ego frame, unnormalized (as sample_candidates gives)
            encoding: optional precomputed DP encoder output, see encode

        Returns:
            logits: [B, K, len(METRICS)]
        """
        memory, mask = self.encode(inputs, encoding)
        plans = (candidates - self.plan_mean) / self.plan_std
        x = self.plan_proj(plans.flatten(start_dim=2))                     # [B, K, D]
        for layer in self.layers:
            x = layer(x, memory, memory_key_padding_mask=mask)
        return self.head(self.norm(x))

    def score(self, inputs: Dict[str, torch.Tensor], candidates: torch.Tensor,
              encoding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[B, K] combined score (CLS formula) of the predicted metric probabilities."""
        return combined_score(torch.sigmoid(self(inputs, candidates, encoding)))

    @classmethod
    def from_dp_checkpoint(cls, args_file: str, ckpt_path: Optional[str], device: str = "cpu", **kwargs) -> "Verifier":
        """Verifier with DP's encoder and route encoder loaded from a DP checkpoint (EMA weights)."""
        config = Config(args_file, guidance_fn=None)
        config.device = device
        model = cls(config, **kwargs)
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["ema_state_dict"]
            state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
            model.load_dp_weights(state_dict)
        return model.to(device)

    def load_dp_weights(self, dp_state_dict: Dict[str, torch.Tensor]) -> None:
        """Copies the encoder and route encoder out of a Diffusion_Planner state dict (no DDP prefix)."""
        route_prefix = "decoder.decoder.dit.route_encoder."
        self.encoder.load_state_dict({k[len("encoder."):]: v for k, v in dp_state_dict.items() if k.startswith("encoder.")})
        self.route_encoder.load_state_dict(
            {k[len(route_prefix):]: v for k, v in dp_state_dict.items() if k.startswith(route_prefix)})

    def save(self, path: str, **extra) -> None:
        torch.save({"state_dict": self.state_dict(), "hparams": self.hparams, **extra}, path)

    @classmethod
    def load(cls, path: str, args_file: str, device: str = "cpu") -> "Verifier":
        """A checkpoint written by save (train.py); args_file is DP's args.json (encoder sizes, normalization)."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls.from_dp_checkpoint(args_file, None, device="cpu", **checkpoint["hparams"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device).eval()
