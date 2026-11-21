# Learned Verifier Architecture

Nov 19, 2026 · @Kaushal

## Overview

The verifier is a 1.46 M-parameter transformer. It scores all K Diffusion Planner (DP) candidates for a scene in one forward pass, replacing PDM+SL as the best-of-K selector. It reuses DP's frozen scene encoder and route encoder. On top of those it trains three things: a per-candidate trajectory embedding, two transformer decoder layers in which candidates attend to each other and to the scene, and one logit per PDM metric. The predicted metrics are combined with nuPlan's CLS formula into one score per candidate, and the highest-scoring candidate is executed.

The goal is to imitate PDM+SL's per-metric labels (`metrics_cv`) at a fraction of its cost. The verifier takes about 3.4 ms and stays flat as K grows; PDM takes about 36 ms at K = 16.

Code: `diffusion_planner/scorer/` on branch `dps-verifier-ablations` (commit `2d333ed`). It covers the model (`model.py`), training (`train.py`), metrics and the combined score (`metrics.py`), and the closed-loop selector (`selectors.py`).

## Inputs

Each sample is one simulation step. It holds the normalized scene tensors DP itself reads, plus K = 16 candidate ego plans sampled from DP. All tensors are in ego's frame, and the shapes below have no batch dimension.

| Input | Shape | Contents | Used by |
| --- | --- | --- | --- |
| `neighbor_agents_past` | 32 × 21 × 11 | 32 nearest agents, 2 s of history at 0.1 s: pose, velocity, box size (first 8 features) and a 3-way type one-hot | Frozen DP encoder |
| `static_objects` | 5 × 10 | 5 nearest static objects: pose, size, type | Frozen DP encoder |
| `lanes` | 70 × 20 × 12 | 70 nearest lanes, 20 points each: centerline (x, y), direction (dx, dy), left and right boundary offsets, traffic-light one-hot (4) | Frozen DP encoder |
| `lanes_speed_limit`, `lanes_has_speed_limit` | 70 × 1 each | Lane speed limit and whether one exists | Frozen DP encoder |
| `route_lanes` (+ speed-limit pair) | 25 × 20 × 12 | Up to 25 lanes on the route | Frozen DP route encoder |
| `ego_current_state` | 10 | x, y, cos, sin, vx, vy, ax, ay, steering, yaw rate | Learned ego token |
| `candidates` | 16 × 80 × 4 | DP plans: (x, y, cos, sin) at 0.1 s for 8 s, unnormalized | Learned plan embedding |

- **Candidates** come from DP's diffusion sampler with initial-noise std 1.5 for all 16 (`noise_scale`). The verifier sees all 80 poses (8 s), but the PDM labels score only the first 40 (4 s).
- **Plan normalization** follows DP's `state_normalizer`: (x − 10) / 20, y / 20, with cos and sin unchanged.
- **Ego state carries no motion.** In the sample I checked, every field after (x, y, cos, sin) is zero. DP's data processor notes that it uses only the pose at inference. So the ego token tells the verifier nothing about speed or acceleration, and current speed is visible only through the plans' first poses. I haven't checked this across the whole dataset.
- **Padding**: an agent, object or lane slot that is all zeros is masked out of cross-attention, using the same rule as DP's encoder. Slot 0 is never masked.

## Model architecture

The verifier is a transformer decoder whose queries are the 16 candidate plans. Its keys and values are 109 scene tokens, most of them from DP's frozen encoders. Of 3.34 M parameters in total, 1.46 M are trained and 1.88 M are frozen.

&#91;embedded content: verifier architecture · frozen DP encoders, trained candidate decoder\]

Data flows top to bottom. The scene path (left) runs once per sample; the candidate path (right) runs once per plan. The two meet only in the decoder's cross-attention.

| Component | What it does | Input → output | Parameters | Trained |
| --- | --- | --- | --- | --- |
| DP scene encoder | MLP-Mixer per entity type (agents depth 3, lanes depth 3 with speed-limit and traffic-light embeddings, static objects), plus a Linear(7→192) pose embedding, then a 3-block self-attention fusion encoder with 6 heads | agents, static objects, lanes → 107 × 192 | 1,799,040 | No (DP EMA weights) |
| DP route encoder | One Mixer block over the 25 × 20 route points, mean-pooled | 25 × 20 × 4 → 1 × 192 | 81,920 | No |
| `scene_proj` | Linear layer applied to the 107 encoder tokens and the route token | 108 × 192 → 108 × 192 | 37,056 | Yes |
| `ego_proj` | MLP 10→192→192 with GELU | 10 → 1 × 192 | 39,168 | Yes |
| `plan_proj` | Normalizes each plan, flattens it, MLP 320→384→192 with GELU | 16 × 80 × 4 → 16 × 192 | 197,184 | Yes |
| Decoder layers (×2) | `nn.TransformerDecoderLayer`, d = 192, 6 heads of 32, FFN 768, GELU, pre-norm, dropout 0.1 | 16 × 192 queries, 109 × 192 memory → 16 × 192 | 1,186,944 | Yes |
| Final LayerNorm + head | LayerNorm, then Linear 192→7 | 16 × 192 → 16 × 7 logits | 1,735 | Yes |

**Details that matter**

- **The order of the candidates doesn't matter.** Plan tokens get no index or position embedding, and self-attention is unmasked. Shuffling the 16 candidates shuffles the outputs the same way. The only effect of order is that ties go to the lowest index.
- **Each candidate sees the others.** Self-attention lets a candidate's prediction depend on the rest of the set. That matters for progress, whose label is normalized by the best candidate in the scene.
- **Each plan is compressed into one token before it meets the scene.** All 80 poses are flattened into one 320-d vector and embedded by the MLP. Cross-attention therefore compares whole-plan summaries against scene tokens, never individual poses against individual lane points.
- **The frozen parts never train.** The DP encoders run under `torch.no_grad()` and stay in eval mode during training, so their dropout and drop-path are off. Their outputs can be cached or reused from DP's sampler.
- **Memory mask.** Empty agent, static and lane slots are masked from cross-attention using DP's own rule. The route and ego tokens are never masked.
- **Ablation switch.** `--no-scene` (`use_scene=False`) drops both DP encoders, leaving the ego token as the only memory. This is the trajectory-only ablation.

## Keys and values in detail

The decoder's keys and values come from 109 memory tokens of 192 dimensions each: one per agent, static object and lane, plus one route token and one ego token. Every entity is summarized into a **single vector** before the verifier sees it. That includes each lane: its 20 points and both boundaries are mean-pooled into one token.

| Tokens | Count | Raw input per entity | How the vector is built | Position embedding (7-d) |
| --- | --- | --- | --- | --- |
| Agents | 32 | 21 timesteps × (x, y, cos, sin, vx, vy, width, length) + a validity flag, and a 3-way type one-hot | Channel MLP 9→128 per timestep; time-mixing MLP 21→64; 3 MLP-Mixer blocks; **mean-pool over time**; + type embedding (3→128); LayerNorm; MLP 128→192→192 | Pose at the latest timestep + type \[1, 0, 0\] |
| Static objects | 5 | (x, y, cos, sin, width, length, 4-way type) | MLP 10→192→192 | Pose + type \[0, 1, 0\] |
| Lanes | 70 | 20 points × (x, y, direction dx dy, left-boundary offset, right-boundary offset); speed limit; traffic light (4-way one-hot, from point 0) | Channel MLP 8→128 per point; point-mixing MLP 20→64; 3 MLP-Mixer blocks; **mean-pool over points**; + speed-limit embedding (Linear 1→128, or a learned "no limit" vector); + traffic-light embedding (4→128); LayerNorm; MLP 128→192→192 | **Midpoint only** (point 10): x, y, heading + type \[0, 0, 1\] |
| Route | 1 | 25 route lanes × 20 points × (x, y, dx, dy); no boundaries, speed limits or lights | All 500 points as one sequence: channel MLP 4→64; point-mixing MLP 500→32; 1 Mixer block; mean-pool; MLP 64→192→192 | none |
| Ego | 1 | 10-d ego state (only the pose is non-zero in practice) | Verifier's own MLP 10→192→192 (trained) | none |

**From entity vectors to memory**

1. **Position embedding.** Each valid agent, static or lane vector gets Linear(7→192) of its position vector added. Empty slots stay zero.
2. **Fusion encoder.** The 107 tokens pass through 3 pre-norm self-attention blocks (6 heads, MLP ratio 4) with empty slots masked, then a final LayerNorm. After this, each token carries context from the others, but it is still one vector per entity.
3. **Verifier projection.** The 107 fused tokens and the route token go through the trained `scene_proj` (Linear 192→192). The ego token is appended without it. Result: memory M of 109 × 192, with a padding mask.

Steps 1–2 and the route encoder are frozen DP weights, run under `no_grad`.

**How the decoder reads the memory.** In each of the 2 decoder layers, cross-attention projects the same memory with that layer's own weights into keys and values. Each of the 6 heads has 32 dimensions:

```latex
K^{(l)} = M W_K^{(l)}, \quad V^{(l)} = M W_V^{(l)}, \quad \text{out}_q = \sum_{j \notin \text{pad}} \operatorname{softmax}_j\!\Big(\tfrac{q \cdot k_j}{\sqrt{32}}\Big)\, v_j
```

The decoder is pre-norm, so LayerNorm applies to the queries only; the memory goes into W\_K and W\_V as it is. Both layers read the same 109 tokens; only the projections differ.

**Why this matters for drivable area.** A query can only receive a weighted average of value vectors, and each lane contributes exactly one. Whatever boundary detail survived the mean-pool over 20 points is all a query can use. The lane's position embedding encodes only its midpoint. Per-timestep queries (next section) make the query side precise, but they still read these same per-lane summaries.

## Proposed: per-timestep queries

This is a proposal, not implemented yet. Each candidate becomes a sequence of 41 query tokens, one per state PDM scores, instead of one flattened token. Each timestep attends to the scene separately, and a failure at a single step can be detected at that step. The keys and values stay the same 109 memory tokens.

&#91;embedded content: proposed per-timestep verifier · query side redesigned, keys and values unchanged\]

One shared decoder processes each candidate independently, with the 16 candidates folded into the batch dimension. The only step that looks across candidates is the last one, which has no learned parameters.

**What changes compared with the current model**

|  | Current | Proposed |
| --- | --- | --- |
| Query tokens | 16, one per candidate | 16 × 41 = 656, one per (candidate, timestep) |
| Plan horizon seen | 80 steps (8 s), flattened into 320 numbers | 41 states (now + 4 s), matching what the label scores |
| Order in time | implicit in the flattened vector | learned time embedding e\_t per step, plus temporal self-attention |
| Self-attention | across the 16 candidates | across the 41 steps of one candidate |
| Cross-attention | one whole-plan query per candidate | one query per pose, so each pose finds the lanes and agents near it |
| Any-step metrics | one logit from the pooled plan | per-step failure probabilities p\_t, combined as 1 − ∏(1 − p\_t) |
| Progress | normalized label predicted directly, needs cross-candidate attention | raw metres per candidate; safety gating and normalization applied after prediction |
| Keys and values | 109 memory tokens | unchanged |

**What it fixes and what it doesn't.** The query side is no longer compressed: a corner leaving the road at step 23 shows up in step 23's query. The keys and values are still one mean-pooled vector per lane, positioned only by its midpoint (previous section). So the precision of the boundary check is limited by what DP's lane summaries keep.

Two additions would address the scene side:

- the optional per-step corner-to-lane distances shown in the diagram;
- raw lane-boundary points (70 × 20) as extra memory tokens.

**Cost.** Cross-attention grows from 16 × 109 to 656 × 109 query-key pairs per sample, plus 16 small 41 × 41 temporal attentions. That should still be well under PDM's 36 ms at K = 16, but it hasn't been measured.

## Outputs and scoring

The head outputs 7 logits per candidate, one per PDM metric. A sigmoid turns each into a probability-like value in \[0, 1\], and the 7 values are combined into one score per candidate with nuPlan's CLS formula. The candidate with the highest score is selected, with ties going to the lowest index as in PDM.

| # | Metric | Kind | Label values | Weight |
| --- | --- | --- | --- | --- |
| 0 | `no_at_fault_collision` | multiplicative | 0, 0.5, 1 | — |
| 1 | `drivable_area` | multiplicative | 0, 1 | — |
| 2 | `driving_direction` | multiplicative | 0, 0.5, 1 | — |
| 3 | `progress` | weighted | \[0, 1\], raw progress ÷ best of the K | 5 |
| 4 | `ttc` | weighted | 0, 1 | 5 |
| 5 | `speed_limit` | weighted | \[0, 1\], nuPlan speed-limit compliance | 4 |
| 6 | `comfort` | weighted | 0, 1 | 2 |

```latex
\text{score} = m_{\text{coll}} \cdot m_{\text{drivable}} \cdot m_{\text{dir}} \cdot \frac{5\,m_{\text{prog}} + 5\,m_{\text{ttc}} + 4\,m_{\text{speed}} + 2\,m_{\text{comfort}}}{16}
```

- The weights are CLS's. PDM-Closed's own score drops the speed-limit term (5 / 5 / 0 / 2), but the labels and the verifier both use CLS weights, because early PDM-selected runs lost 2–3 points of speed-limit compliance.
- Progress is normalized **within the scene**: each candidate's raw metres along the centerline divided by the best of the 16. So the progress label is relative, and the verifier has to infer the scene's best progress from the other candidates. Self-attention between candidates makes that possible.
- **Classifier variant** (`--objective classify`): the head outputs 1 logit per candidate, used directly as the score, with no metric heads and no CLS formula. It only changes the head from 192→7 to 192→1.

## Training

The verifier is trained to reproduce PDM+SL's per-metric labels (`--labels cv`). The loss is per-metric BCE plus a listwise ranking loss on the combined score, weighted 3×. Only the 1.46 M new parameters are trained; DP's encoders stay frozen and always run in eval mode.

**Labels.** `LabelRecorder` drives non-reactive closed-loop simulation with the PDM selector, so the recorded states are the ones a deployed selector reaches. At every recorded step it scores the same 16 candidates twice:

- `metrics_cv`: PDM with constant-velocity forecasts of other agents, which is exactly what PDM+SL sees at test time. **This is the training and evaluation target.**
- `metrics_oracle`: PDM with the logged futures of other agents. This is exact in non-reactive simulation but invalid in reactive simulation, and is now only a secondary report.

The dataset is `train-labels-2k`: 1,992 scenarios with 30 samples each, 59,760 samples in total. The split is by scenario, never by sample, because samples from one scenario are nearly identical. 10% of scenarios go to validation with `--split-seed 0`, giving 53,790 training samples from 1,793 scenarios and 5,970 validation samples from 199.

**Loss** (`--objective metrics`, the default):

```latex
\mathcal{L} = \text{BCE}\big(\text{logits}, m\big) + 3 \cdot \Big(-\sum_{k=1}^{K} \operatorname{softmax}_k\!\big(S(m)/\tau\big) \, \log \operatorname{softmax}_k\!\big(S(\sigma(\text{logits}))/\tau\big)\Big), \quad \tau = 0.02
```

- BCE is averaged over all 16 × 7 outputs with soft targets, so the 0.5 labels for collision and direction are used as they are. Every metric gets equal weight, even though the multiplicative ones decide most of the regret.
- S is the CLS combined score. The ranking term is cross-entropy between the softmax over the 16 candidates of the target scores and of the predicted scores. With τ = 0.02 the target distribution is close to one-hot on the best candidate.
- **Classifier objective** (`--objective classify`): cross-entropy between softmax(head output) and softmax(S(m) / 0.02), computed only on samples where the candidates are not all tied (about 78% under cv labels).

**Optimization**

| Setting | Value |
| --- | --- |
| Optimizer | AdamW, lr 3e-4, weight decay 0.01 |
| Schedule | OneCycle, 10% warm-up, over all steps |
| Batch size | 64 samples (64 × 16 candidates), about 841 steps per epoch |
| Epochs | 30 for w5\_cv (script default is 20) |
| Gradient clipping | global norm 1.0 |
| Dropout | 0.1 in the decoder layers (0 for overfit checks) |
| Checkpoint kept | the epoch with the lowest validation regret (`best.pt`), plus `last.pt` |
| Time | about 42 s per epoch on the RTX 5080 |

The best checkpoint is chosen on the same validation set that results are reported on. Over 30 epochs this biases the reported regret slightly downward.

## Evaluation and inference

The verifier is judged on which candidate it picks, not on how well it predicts each metric. The headline numbers are regret and top-1 against the target combined score, compared with three reference selectors.

**Selection metrics** (`train.py`, every epoch, on the 199 validation scenarios):

- **Regret**: the mean of (best target score among the 16) − (target score of the pick).
- **Top-1**: the share of samples where the pick is among the best candidates, with ties counted as hits.
- **Reference selectors**: First (candidate 0), Random (expected value), and PDM (argmax of `metrics_cv`, which scores 0 regret by construction under cv labels).
- **Informative subset**: the same metrics restricted to samples whose candidates are not all tied (78% of validation samples under cv labels).

**Diagnostics** (`scripts/diagnose_verifier.py`): per-metric MAE, Brier score, global failure AUC and within-scene pair accuracy. They also assign each miss a primary cause and break regret down by scenario type.

**Inference in closed loop.** `VerifierSelector` (`selectors.py`) runs the model on the 16 candidates and picks the highest score, with ties going to the lowest index. In simulation it is enabled with `SELECTOR=verifier` in `scripts/run_sim.sh`, and the checkpoint is set by `VERIFIER_CKPT`. The default checkpoint is still `results/verifier/w3_oracle/best.pt`, an oracle-label model, so a closed-loop run of a cv-trained verifier has to set it explicitly.

**Runtime** (`results/runtime_bench.csv` and `runtime_k_sweep.csv`, RTX 5080):

| Selector | K = 4 | K = 16 | K = 64 |
| --- | --- | --- | --- |
| Verifier, including the DP encoder pass | 3.4 ms | 3.4 ms | 3.6 ms |
| Verifier, reusing the sampler's encoder output | 1.1 ms | 1.1 ms | 1.2 ms |
| PDM selection | 18.5 ms | 35.7 ms | 113.5 ms |
| DP sampling of the K candidates (for scale) | 25.5 ms | 26.4 ms | 30.2 ms |

`Verifier.forward` accepts the encoder output that DP's sampler has already computed (`encoding=`), but `VerifierSelector` doesn't pass it in. Wiring that through would save about 2.3 ms per step.

## Current results and known limitations

The w5\_cv run (3 seeds, cv labels, 30 epochs) reaches a validation regret of 0.0370 ± 0.0007 and a top-1 of 0.834. Random selection scores 0.0625 regret and PDM+SL scores 0, so the verifier closes about 41% of that gap. Most of what remains is the verifier failing to rank candidates within a scene on drivable area and collision.

| Seed | Best epoch | Regret | Top-1 |
| --- | --- | --- | --- |
| 0 | 18 | 0.0376 | 0.834 |
| 1 | 18 | 0.0360 | 0.832 |
| 2 | 23 | 0.0375 | 0.835 |

The limitations below follow from the architecture:

- **Safety metrics are ranked near chance within a scene.** Within-scene pair accuracy is 0.58 for drivable area, 0.57 for collision and 0.53 for driving direction, against 0.93 for progress. Drivable area and collision cause 73% of regret.
- **One pooled vector per plan.** `plan_proj` flattens all 80 × 4 states into a single 192-d token. Nothing in the model compares individual poses against individual lane points, but the drivable-area label depends on one corner at one timestep.
- **The plan is longer than what is scored.** The verifier sees 8 s of plan; the labels score the LQR-tracked first 4 s. Neither the horizon nor the tracker's deviation is given to the model.
- **Missing map information.** DP's lane inputs carry no car-park polygons and only the 70 nearest lanes. PDM's drivable map includes car parks within 50 m.
- **No ego motion state.** The ego token's velocity, acceleration, steering and yaw-rate fields are zero, so speed comes only from the plans' first poses.
- **Frozen scene features.** DP's encoder was trained for planning, not for checking boundaries. With it frozen, the only scene-specific learning happens in the 192→192 `scene_proj` and in cross-attention.
- **Equal-weight BCE.** All 7 metrics get the same loss weight, although the three multiplicative ones decide most of the regret.
- **Never run in closed loop with cv labels.** The closed-loop default checkpoint is still the oracle-label `w3_oracle`.

## Planned ablations

- [ ] **Drop the ego token.**

  The ego input is the same constant in every sample (`[-0.5, 0, 1, 0, …, 0]`; DP zeroes the dynamic fields), so the token carries no scene information. It may still act as an attention sink: a constant slot where queries with nothing relevant to attend to can put their weight. Run: add a `--no-ego-token` flag that leaves it out of the memory (108 tokens; not allowed together with `--no-scene`), then 3 seeds on cv labels with the w5\_cv setup (`--split-seed 0`, 30 epochs, about 70 min on aegis). Keep the token if regret rises clearly above the w5\_cv baseline of 0.0370 ± 0.0007; drop it if regret stays within that range.

* [ ] **Without self-attention.**

  Each decoder layer keeps cross-attention to the scene but drops self-attention. The two archs ask different questions. In the plan arch (w5), self-attention is the only path by which the 16 candidates see each other, so this tests whether comparing candidates helps ranking. In the timestep arch (w6), self-attention mixes the 41 steps of one plan, so this tests whether each step can be judged from its own state and the scene. Run: add a `--no-self-attn` flag. The simplest version passes a diagonal `tgt_mask` so each token attends only to itself, which keeps the parameter count. Then train 3 seeds per arch on cv labels with `--split-seed 0` and 30 epochs. Compare against w5\_cv (0.0370 ± 0.0007) and w6\_timestep (0.0352 ± 0.0013). Keep self-attention in an arch if dropping it raises regret beyond that arch's range.

- [ ] **Mark route lanes on the lane tokens and drop the route token.** Low priority: it removes only one token and isn't expected to gain much.

  The 70 lane tokens don't say which lanes are on the route; the only route signal is the single pooled route token. Run: add a learned "on route" embedding to each lane token whose lane is on the route (DP's map processing already computes this per lane, `map_process.py:370`, but it isn't passed on as a feature), and leave the route token out of the memory. Train 3 seeds with the w5\_cv setup, about 70 min on aegis. Keep the change if regret stays within 0.0370 ± 0.0007, or if `changing_lane` regret (currently 0.069) improves.
