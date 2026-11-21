"""
Learned verifier for best-of-K Diffusion Planner (task 2.11).

Scene: DP's frozen encoder (agents, static objects, lanes: 107 tokens x 192) plus DP's frozen route encoder
(1 token) and a learned ego-state token. Two candidate architectures (arch):

  plan      one token per ego plan, an MLP over its normalized 80 x 4 states. Transformer decoder layers:
            candidates attend to each other (self-attention) and to the scene (cross-attention). Heads: one logit
            per metric of diffusion_planner.scorer.metrics.
  timestep  one token per scored state of each plan (current state + the 40 poses PDM scores, 4 s at 0.1 s): pose,
            velocity, acceleration, plus a learned time embedding. One shared decoder per candidate: the states of
            a plan attend to each other (self-attention) and each state to the scene (cross-attention); candidates
            never see each other. Metrics from per-step outputs: the any-step metrics (collision, drivable area,
            driving direction, TTC) fail with logit logsumexp_t(l_t), a smooth max over per-step failure logits
            (a product of per-step pass probabilities starves the per-step logits of gradient), speed limit
            as the mean per-step compliance, comfort by attention pooling, progress as raw metres from the last
            state, gated by the predicted multiplicative metrics and normalized by the best candidate as PDM does.

Both return one logit per metric, trained with BCE against the PDM labels; the combined score uses CLS's formula.

lane_segments > 0 (train.py --lane-segments) adds trainable lane segment tokens to the memory: each of DP's lane inputs
(20 points) is cut into lane_segments pieces and each piece becomes its own token, placed at its own center. DP's
frozen lane encoder mean-pools a lane's 20 points into one token, which loses the boundary geometry the drivable-area
label depends on; these tokens keep it (see LaneSegmentEncoder).

    model = Verifier.from_dp_checkpoint("checkpoints/args.json", "checkpoints/model.pth")
    scores = model.score(inputs, candidates)   # [B, K], higher is better
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp

from diffusion_planner.model.diffusion_planner import Diffusion_Planner_Encoder
from diffusion_planner.model.module.decoder import RouteEncoder
from diffusion_planner.scorer.metrics import METRICS, combined_score
from diffusion_planner.utils.config import Config

ARCHS = ("plan", "timestep")
HORIZON = 40               # plan poses PDM scores (4 s at 0.1 s); the timestep arch adds the current state
DT = 0.1                   # [s] between plan poses
ANY_STEP = [0, 1, 2, 4]    # METRICS a candidate fails if any scored state fails: collision, drivable, direction, ttc
SPEED_LIMIT, COMFORT, PROGRESS = 5, 6, 3
STEP_FEATURES = 9          # normalized pose (4), velocity (2), acceleration (2), speed (1)
PROGRESS_SCALE = 10.0      # [m] unit of the raw progress head and its loss
PROGRESS_THRESHOLD = 0.1   # [m] PDM's PROGRESS_DISTANCE_THRESHOLD: below it every safe candidate gets progress 1
PROB_EPS = 1e-4            # probabilities are clamped to [eps, 1 - eps] before turning them back into logits
GEOM_OUTPUTS = 5           # geom_head per step: off-lane logit per corner (4), then the overshoot [m]


class LaneSegmentEncoder(nn.Module):
    """
    Trainable lane segment tokens. Each lane's points are cut into `segments` consecutive pieces; a token per piece
    from its local shape (centerline, direction and boundary offsets of its points, relative to the piece's center)
    plus the piece's position (center x, y and heading). No pooling over a lane, no traffic light or speed limit
    (DP's own lane tokens still carry those).
    """

    def __init__(self, lane_len: int, segments: int, dim: int):
        super().__init__()
        assert lane_len % segments == 0, f"lane_len {lane_len} must split into {segments} equal segments"
        self.segments, self.points = segments, lane_len // segments
        self.shape_proj = Mlp(in_features=self.points * 8, hidden_features=dim, out_features=dim, act_layer=nn.GELU)
        self.pos_proj = nn.Linear(4, dim)

    def forward(self, lanes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lanes: [B, L, N, 12] normalized DP lane inputs (x, y, dx, dy, left - center (2), right - center (2),
                   traffic light (4)); padded points are all zero

        Returns:
            tokens: [B, L * segments, D]; mask: [B, L * segments] bool, True = empty segment
        """
        B, L = lanes.shape[:2]
        x = lanes[..., :8].reshape(B, L * self.segments, self.points, 8)   # [B, S, P, 8]
        valid = (x != 0).any(dim=-1)                                       # [B, S, P]
        mask = ~valid.any(dim=-1)                                          # [B, S]
        count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
        center = (x[..., :2] * valid[..., None]).sum(dim=2) / count        # [B, S, 2], mean of the valid points
        direction = (x[..., 2:4] * valid[..., None]).sum(dim=2)
        heading = torch.atan2(direction[..., 1], direction[..., 0])
        local = torch.cat([x[..., :2] - center[:, :, None], x[..., 2:]], dim=-1) * valid[..., None]
        pos = torch.cat([center, torch.cos(heading)[..., None], torch.sin(heading)[..., None]], dim=-1)
        tokens = self.shape_proj(local.flatten(start_dim=2)) + self.pos_proj(pos)
        return tokens * (~mask)[..., None], mask


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

    def __init__(self, config, depth: int = 2, num_heads: int = 6, dropout: float = 0.1, use_scene: bool = True,
                 num_outputs: int = len(METRICS), arch: str = "plan", geom_head: bool = False,
                 lane_segments: int = 0):
        """
        Args:
            config: DP config (diffusion_planner.utils.config.Config or the training args)
            depth: transformer decoder layers over the candidates (plan) or over each candidate's states (timestep)
            use_scene: False drops the encoder and route tokens (trajectory-only ablation, task 3.3); the ego-state
                       token stays
            num_outputs: len(METRICS) = one logit per metric (combined with CLS's formula); 1 = a direct score per
                         candidate with no metric heads (classifier, train.py --objective classify; plan arch only)
            arch: "plan" (one token per candidate) or "timestep" (one token per scored state), see the module docstring
            geom_head: timestep arch: a per-step head for the dense analytic drivable-area labels (train.py
                       --analytic-weight): one off-lane logit per corner of ego's box, then the overshoot [m]
            lane_segments: > 0 adds lane_num * lane_segments trainable lane segment tokens to the memory
                           (LaneSegmentEncoder); 0 = DP's tokens only (w6). Needs use_scene
        """
        super().__init__()
        dim = config.hidden_dim
        assert num_outputs in (1, len(METRICS)), f"num_outputs must be 1 or {len(METRICS)}"
        assert arch in ARCHS, f"arch must be one of {ARCHS}"
        assert arch == "plan" or num_outputs == len(METRICS), "the timestep arch predicts the metrics (no classifier)"
        self.hparams = {"depth": depth, "num_heads": num_heads, "dropout": dropout, "use_scene": use_scene,
                        "num_outputs": num_outputs, "arch": arch, "geom_head": geom_head,
                        "lane_segments": lane_segments}
        self.use_scene = use_scene
        self.arch = arch

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
        assert lane_segments == 0 or use_scene, "lane segment tokens are scene tokens (need use_scene)"
        self.lane_segments = LaneSegmentEncoder(config.lane_len, lane_segments, dim) if lane_segments else None
        self.ego_proj = Mlp(in_features=10, hidden_features=dim, out_features=dim, act_layer=nn.GELU)
        if arch == "plan":
            self.plan_proj = Mlp(in_features=config.future_len * 4, hidden_features=2 * dim, out_features=dim,
                                 act_layer=nn.GELU)
        else:
            self.step_proj = Mlp(in_features=STEP_FEATURES, hidden_features=2 * dim, out_features=dim,
                                 act_layer=nn.GELU)
            self.time_emb = nn.Parameter(torch.randn(HORIZON + 1, dim) * 0.02)

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(dim, num_heads, dim_feedforward=4 * dim, dropout=dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        if arch == "plan":
            self.head = nn.Linear(dim, num_outputs)
        else:
            # Per step: a failure logit per any-step metric, then a speed-limit compliance logit. At logit 0 on
            # every step a candidate starts at failure logit log(41), P(pass) ~ 0.98 (see _timestep_forward)
            self.step_head = nn.Linear(dim, len(ANY_STEP) + 1)
            self.comfort_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.comfort_pool = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            self.comfort_head = nn.Linear(dim, 1)
            self.progress_head = nn.Linear(dim, 1)
            self.geom_head = nn.Linear(dim, GEOM_OUTPUTS) if geom_head else None

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
        if self.lane_segments is not None:
            segments, segment_mask = self.lane_segments(inputs["lanes"].float())
            memory = torch.cat([memory, segments], dim=1)
            mask = torch.cat([mask, segment_mask], dim=1)
        return memory, mask

    def step_features(self, candidates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            candidates: [B, K, T, 4] ego plans (x, y, cos, sin), ego frame, unnormalized, T >= HORIZON

        Returns:
            features: [B, K, HORIZON + 1, STEP_FEATURES] per scored state (current state first, ego at the origin):
                      normalized pose, velocity / 10, acceleration / 5 (clamped to +-5), speed / 10, from finite
                      differences of the poses (the first state copies the second's velocity and acceleration)
        """
        plans = candidates[:, :, :HORIZON]
        current = plans.new_tensor([0.0, 0.0, 1.0, 0.0]).expand(*plans.shape[:2], 1, 4)
        states = torch.cat([current, plans], dim=2)                        # [B, K, H + 1, 4]
        vel = torch.diff(states[..., :2], dim=2) / DT
        vel = torch.cat([vel[:, :, :1], vel], dim=2)
        acc = torch.diff(vel, dim=2) / DT
        acc = torch.cat([acc[:, :, :1], acc], dim=2)
        pose = (states - self.plan_mean) / self.plan_std
        speed = vel.norm(dim=-1, keepdim=True)
        return torch.cat([pose, vel / 10.0, (acc / 5.0).clamp(-5.0, 5.0), speed / 10.0], dim=-1)

    def _timestep_forward(self, memory: torch.Tensor, mask: torch.Tensor, candidates: torch.Tensor
                          ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """The timestep arch (see the module docstring): logits [B, K, len(METRICS)], aux {"progress_raw": [B, K]}."""
        B, K = candidates.shape[:2]
        x = self.step_proj(self.step_features(candidates)) + self.time_emb # [B, K, H + 1, D]
        x = x.flatten(0, 1)                                                # [B * K, H + 1, D]
        memory, mask = memory.repeat_interleave(K, dim=0), mask.repeat_interleave(K, dim=0)
        for layer in self.layers:
            x = layer(x, memory, memory_key_padding_mask=mask)
        x = self.norm(x)

        step = self.step_head(x)                                           # [B * K, H + 1, len(ANY_STEP) + 1]
        # Any-step metrics: the candidate fails with logit logsumexp_t(l_t), a smooth max over its states (one
        # failing state suffices; the softmax-weighted gradient reaches every step). Pass logit = its negation
        any_step = -torch.logsumexp(step[..., :len(ANY_STEP)], dim=1)     # [B * K, len(ANY_STEP)]
        speed = torch.logit(torch.sigmoid(step[..., -1]).mean(dim=1), eps=PROB_EPS)
        pooled, _ = self.comfort_pool(self.comfort_query.expand(len(x), -1, -1), x, x)
        comfort = self.comfort_head(pooled[:, 0])[:, 0]
        raw = F.softplus(self.progress_head(x[:, -1])[:, 0]) * PROGRESS_SCALE

        logits = x.new_zeros(B * K, len(METRICS))
        logits[:, ANY_STEP] = any_step
        logits[:, SPEED_LIMIT] = speed
        logits[:, COMFORT] = comfort
        logits = logits.view(B, K, -1)
        raw = raw.view(B, K)

        # Progress as PDM normalizes it: raw progress gated by the multiplicative metrics, over the best candidate
        safe = torch.sigmoid(logits[..., :3]).prod(dim=-1)                 # collision * drivable * direction
        gated = raw * safe
        best = gated.max(dim=1, keepdim=True).values
        progress = torch.where(best > PROGRESS_THRESHOLD, gated / best.clamp_min(PROGRESS_THRESHOLD), safe)
        logits = torch.cat([logits[..., :PROGRESS], torch.logit(progress, eps=PROB_EPS)[..., None],
                            logits[..., PROGRESS + 1:]], dim=-1)
        aux = {"progress_raw": raw, "step_logits": step.view(B, K, *step.shape[1:])}
        if self.geom_head is not None:
            aux["geom"] = self.geom_head(x).view(B, K, x.shape[1], GEOM_OUTPUTS)
        return logits, aux

    def forward(self, inputs: Dict[str, torch.Tensor], candidates: torch.Tensor,
                encoding: Optional[torch.Tensor] = None, return_aux: bool = False):
        """
        Args:
            inputs: normalized model inputs, each [B, ...]
            candidates: [B, K, T, 4] ego plans (x, y, cos, sin), ego frame, unnormalized (as sample_candidates gives)
            encoding: optional precomputed DP encoder output, see encode
            return_aux: also return a dict of auxiliary outputs (timestep arch: "progress_raw" [B, K] metres)

        Returns:
            logits: [B, K, num_outputs] (len(METRICS) metric logits, or 1 direct score); (logits, aux) with return_aux
        """
        memory, mask = self.encode(inputs, encoding)
        if self.arch == "timestep":
            logits, aux = self._timestep_forward(memory, mask, candidates)
        else:
            plans = (candidates - self.plan_mean) / self.plan_std
            x = self.plan_proj(plans.flatten(start_dim=2))                 # [B, K, D]
            for layer in self.layers:
                x = layer(x, memory, memory_key_padding_mask=mask)
            logits, aux = self.head(self.norm(x)), {}
        return (logits, aux) if return_aux else logits

    def score(self, inputs: Dict[str, torch.Tensor], candidates: torch.Tensor,
              encoding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[B, K] higher is better: the combined score (CLS formula) of the predicted metric probabilities, or the
        direct score of a classifier (num_outputs = 1)."""
        out = self(inputs, candidates, encoding)
        if out.shape[-1] == 1:
            return out[..., 0]
        return combined_score(torch.sigmoid(out))

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
