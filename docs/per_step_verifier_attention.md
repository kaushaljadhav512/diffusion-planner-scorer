# Per-Step Verifier: Multi-Head Attention

Nov 20, 2026 · @Kaushal

## Overview

The per-step verifier (`--arch timestep`, run w6\_timestep) uses multi-head attention in three places. Each of the 16 candidates becomes a sequence of 41 state tokens. Two decoder layers let those tokens attend to each other (self-attention) and to 109 frozen scene tokens (cross-attention). A learned query then pools the 41 outputs into one vector (attention pooling). All attention uses width D = 192 and 6 heads of 32 dims each. The code is in `diffusion_planner/scorer/model.py` (`Verifier`, `_timestep_forward`).

| Tensor | Shape (batch 64) | Meaning |
| --- | --- | --- |
| Step queries | \[B·K, 41, 192\] = \[1024, 41, 192\] | one sequence per candidate: current state plus 40 PDM-scored poses |
| Scene memory | \[B·K, 109, 192\] | DP encoder's 107 tokens, the route token and the ego token, copied K times |
| Memory padding mask | \[B·K, 109\] bool | True marks an empty agent, static-object or lane slot |
| Decoder output | \[B·K, 41, 192\] | one context-aware vector per state |
| Pooled vector | \[B·K, 1, 192\] | comfort head input (w6), or the single score's input (w7) |

Candidates never attend to each other. The batch is flattened to B·K sequences, so the model judges each plan on its own against the scene. This is the main structural difference from the plan arch (w5), where the 16 plan tokens use self-attention across candidates.

## Inputs and tokens

Attention only sees what the tokens encode. The queries carry the plan's kinematics and time. The keys and values carry the scene as DP's frozen encoder summarizes it.

**Queries: 41 step tokens per candidate** (`step_features`, `step_proj`, `time_emb`)

- Step 0 is the current state (ego at the origin). Steps 1 to 40 are the plan poses PDM scores: 4 s at 0.1 s.
- Each step has 9 features. Pose (x, y, cos, sin) uses DP's state normalizer. Velocity / 10 and acceleration / 5 (clamped to ±5) come from finite differences, plus speed / 10.
- An MLP (9 → 384 → 192, GELU) embeds each step, and a learned time embedding of shape \[41, 192\] is added. This is the only positional signal for self-attention. Without it, step 12 and step 30 at the same pose would look alike.

**Keys and values: 109 scene memory tokens** (`encode`)

| Slots | Count | What one token is | Position it carries |
| --- | --- | --- | --- |
| Agents | 32 | one neighbour's past track, through an MLP-mixer | its latest pose plus a type one-hot |
| Static objects | 5 | one object's state | its pose plus a type one-hot |
| Lanes | 70 | one lane polyline mixed into a single vector, with speed limit and traffic light | the midpoint point and heading only |
| Route | 1 | DP's route encoder over the route lanes | none beyond its contents |
| Ego | 1 | `ego_proj` MLP over `ego_current_state` (10 values) | none |

- The first 107 tokens come from DP's encoder, after its own fusion self-attention. They have no gradient. The 107 tokens and the route token then pass through one trainable `scene_proj` (192 → 192). That is the only trainable step on the scene side.
- Empty slots are flagged in `scene_padding_mask` and masked as keys. Slot 0 is never masked, so a row is never all padding. Route and ego are never masked.
- The memory is copied K = 16 times (`repeat_interleave`) so each candidate's sequence can attend to it independently.

## Multi-head attention math

Every attention in the model is PyTorch's `nn.MultiheadAttention` with d\_model = 192, h = 6 heads and d\_k = 192 / 6 = 32 per head. Self-attention and cross-attention differ only in where the keys and values come from.

**1. Project.** For a query sequence X (n tokens) and a key/value source M (m tokens):

```latex
Q = X W^Q + b^Q,\qquad K = M W^K + b^K,\qquad V = M W^V + b^V,\qquad W^Q, W^K, W^V \in \mathbb{R}^{192 \times 192}
```

In self-attention M = X (n = m = 41). In cross-attention M is the scene memory (n = 41, m = 109).

**2. Split into heads.** Q, K and V are reshaped from \[n, 192\] to \[6, n, 32\]. Head i owns columns 32i to 32i + 31 of each projection. So the heads are learned subspaces, not separate copies of the input.

**3. Scaled dot-product per head.**

```latex
\mathrm{head}_i = \mathrm{softmax}\!\left(\frac{Q_i K_i^{\top}}{\sqrt{32}} + B\right) V_i
```

The score matrix is 41 × 41 in self-attention and 41 × 109 in cross-attention. Dividing by √32 ≈ 5.66 keeps the logits' variance near 1, so the softmax does not saturate at initialization. The mask B is 0 for valid keys and −∞ for padded ones, so empty slots get exactly zero weight.

**4. Concatenate and mix.**

```latex
\mathrm{MHA}(X, M) = [\mathrm{head}_1; \dots; \mathrm{head}_6]\, W^O + b^O,\qquad W^O \in \mathbb{R}^{192 \times 192}
```

W^O is the only place the six heads' outputs interact.

**Masks in this model**

- Self-attention over steps has no mask. It is bidirectional: step 5 can attend to step 40. That suits scoring, because a plan is judged as a whole and nothing is generated step by step.
- Cross-attention uses `memory_key_padding_mask`, which masks empty agent, static-object and lane slots.
- Pooling (comfort in w6, the score in w7) has no mask, since all 41 steps are always valid.

**Size and cost**

| Quantity | Value |
| --- | --- |
| Parameters per MHA block (4 × 192² weights + 4 × 192 biases) | 148,224 |
| MHA blocks in the timestep arch | 5: 2 self, 2 cross, 1 pool |
| Attention weights per layer, batch 64 (1,024 sequences × 6 heads) | self 41 × 41, cross 41 × 109 |
| Dropout on attention weights | 0.1 (train.py default) |

## Attention blocks in the model

Each of the 2 layers is a PyTorch `nn.TransformerDecoderLayer` with `norm_first=True`. Self-attention runs first, then cross-attention, then a feed-forward block. Each sub-block is wrapped as x + f(LayerNorm(x)).

&#91;embedded content: one candidate through the timestep verifier · 2 decoder layers, 3 attention blocks\]

Shaded boxes are the multi-head attentions. A candidate's 41 steps attend only to each other and to their own copy of the 109 scene tokens. They never see the other 15 candidates. Pre-norm keeps a direct residual path from the step embeddings to the heads.

## From attended tokens to scores

After the two decoder layers and a final LayerNorm, every candidate has a matrix X of 41 × 192: one context-aware vector per state. w6 turns that matrix into 7 metric logits through four different readouts, chosen so each matches how PDM computes that metric. w7 replaces all four with one attention pool and one score.

| Metric (PDM type) | Readout from the 41 step vectors | Matches PDM because |
| --- | --- | --- |
| Collision, drivable area, driving direction (multiplicative) | per-step failure logit, then a smooth max over steps | a plan fails if any scored state fails |
| TTC (weight 5) | same smooth max | same any-step rule |
| Speed limit (weight 4) | per-step compliance, mean over steps | nuPlan's score is 1 − mean overspeed / 2.23 m/s |
| Comfort (weight 2) | attention pooling over all steps | a property of the whole trajectory |
| Progress (weight 5) | raw metres from the last step, gated and divided by the best candidate | PDM's normalization over the K candidates |

### 0. How PDM computes each label

The labels come from tuplan\_garage's `PDMScorer` (`pdm_scorer.py`), called through our `PDMCandidateScorer` (`diffusion_planner/scorer/pdm_scoring.py`). Before any metric runs, PDM does three things the verifier never sees:

- **Tracks the plan.** Each candidate's 40 poses go through PDM's LQR tracker and kinematic bicycle model. The metrics score the simulated states (pose, velocity, acceleration), not the raw plan. The verifier sees only finite differences of the raw poses.
- **Forecasts the agents.** For cv labels, every current detection moves at constant velocity, giving one occupancy map per future step. Objects ego already overlaps at t = 0 are ignored for the whole horizon.
- **Builds ego's footprint.** Ego's 4 corners and center come from the real vehicle size at each of the 41 states.

It also computes three per-step flags that several metrics share (`_calculate_ego_area`):

| Flag | True at a step when |
| --- | --- |
| Non-drivable area | at least one of ego's 4 corners is outside every drivable polygon (lanes, lane connectors, car parks) |
| Multiple lanes | corners touch more than one polygon and no single polygon holds all 4 corners |
| Oncoming traffic | ego's center is in no on-route drivable polygon |

**No at-fault collision** (multiplicative; 1, 0.5 or 0)

- At each step, PDM intersects ego's footprint polygon with every object polygon in that step's occupancy map.
- A collision counts as at fault if nuPlan classifies it as an active front collision or a collision with a stopped track. A lateral collision also counts if ego is in multiple lanes or non-drivable area at that step. Other contacts (being rear-ended, for example) mark that object as ignored from then on.
- An at-fault hit on an agent (vehicle, pedestrian, bicycle) gives 0. A hit on a static object gives 0.5. The candidate keeps its worst value over all steps.
- **Verifier gap:** the fault rules depend on relative heading and whether the agent is moving, which the verifier must infer from past tracks. The 0.5 for static objects is a soft label, while the smooth max is a binary any-step rule.

**Drivable area compliance** (multiplicative; 1 or 0)

- The value is 0 if the non-drivable flag is true at any of the 41 steps, so a single corner off the map at one step fails the plan. There is no tolerance margin.
- **Verifier gap:** this is a pure geometry test of 4 corner points against polygons. The verifier has one vector per lane and no car parks, which is why this metric is stuck.

**Driving direction compliance** (multiplicative; 1, 0.5 or 0)

- PDM sums the distance ego's center travels within each contiguous stretch of steps flagged oncoming traffic. It takes the longest stretch, d\_max.
- d\_max < 2 m gives 1, 2 to 6 m gives 0.5, and 6 m or more gives 0.
- **Verifier gap:** this is a distance threshold, not an any-step rule. A brief 1.5 m excursion off the route lanes still scores 1. The smooth max over per-step logits has to learn that a few flagged steps are fine as long as the plan doesn't travel far there.

**Progress** (weight 5; 0 to 1)

- Raw progress is the distance along the route centerline between ego's center at step 0 and at step 40, clipped at 0.
- Raw progress is multiplied by the product of the three multiplicative metrics, then divided by the largest such value among the 16 candidates.
- If that largest value is 0.1 m or less (for example, when ego is stopped), every candidate with a non-zero product gets 1 and the others get 0.
- **Verifier gap:** none in structure. Our readout copies these steps, using predicted probabilities in place of the 0/0.5/1 values.

**TTC** (weight 5; 1 or 0)

- At each step t, PDM projects ego's footprint forward along its heading at its current speed, by 0, 0.3, 0.6 and 0.9 s. It intersects each projection with the occupancy map at t plus that offset.
- A hit fails the candidate if the object is ahead of ego. It also fails if ego is in multiple lanes, in non-drivable area or in an intersection, and the object is not behind. Steps where ego's speed is below 0.005 m/s are skipped.
- **Verifier gap:** this needs ego's tracked speed and heading, plus where agents will be up to 0.9 s past each step. The per-step queries help here, which matches the w6 gain on TTC.

**Speed limit compliance** (weight 4 in our labels; 0 to 1)

- PDM itself leaves this out. We compute it in `PDMCandidateScorer.speed_limit_compliance` to match nuPlan's CLS.
- At each of the 40 future steps, the limit is the highest limit among the lanes containing ego's center. A lane connector takes the highest limit of its incoming and outgoing lanes. Car parks and lanes without a limit count as no violation.
- The score is max(0, 1 − mean overspeed / 2.23 m/s), averaged over the 40 steps.
- **Verifier gap:** the lane tokens carry a speed limit, but the verifier has to match each step to its lane through attention.

**Comfort** (weight 2; 1 or 0)

- PDM smooths the tracked states with Savitzky–Golay filters, then checks 6 bounds at every step. Comfort is 1 only if all bounds hold at all 41 steps.

| Quantity | Bound |
| --- | --- |
| Longitudinal acceleration | −4.05 to 2.40 m/s² |
| Lateral acceleration | \|a\| ≤ 4.89 m/s² |
| Jerk magnitude | ≤ 8.37 m/s³ |
| Longitudinal jerk | \|j\| ≤ 4.13 m/s³ |
| Yaw rate | ≤ 0.95 rad/s |
| Yaw acceleration | ≤ 1.93 rad/s² |

- **Verifier gap:** this is an any-step rule, but the verifier reads it with attention pooling. The bounds apply to LQR-tracked, filtered states, while the verifier's acceleration comes from raw finite differences (clamped at ±25 m/s² after scaling). Yaw rate and jerk are not given as features and must be learned.

**Combined score.** PDM itself uses multiplicative product × (5 progress + 5 TTC + 2 comfort) / 12. Our labels add speed limit with CLS's weight of 4, for a denominator of 16 (section 6).

### 1. Per-step head

`step_head` is one linear layer, 192 → 5, applied to each of the 41 vectors. It gives 4 failure logits per step (collision, drivable area, driving direction, TTC) and 1 speed-limit compliance logit. The same weights score every step. The time embedding added at the input is what lets the head treat step 3 and step 38 differently.

### 2. Any-step metrics: smooth max

PDM fails a candidate if any of its states fails. The exact probabilistic form is P(pass) = ∏\_t (1 − p\_t). The code says a product of per-step pass probabilities starves the per-step logits of gradient, so it uses logsumexp, a smooth max, on the failure logits l\_t instead:

```latex
\text{fail logit} = \log \sum_{t=0}^{40} e^{\,l_t},\qquad \text{pass logit} = -\text{fail logit}
```

It is bounded by the hard max: max\_t l\_t ≤ logsumexp ≤ max\_t l\_t + log 41, where log 41 ≈ 3.71. Its gradient with respect to each step is that step's softmax weight:

```latex
\frac{\partial\, \text{fail logit}}{\partial l_t} = \frac{e^{\,l_t}}{\sum_s e^{\,l_s}}
```

So most of the learning signal goes to the worst step, but every step gets some. A hard max would train only one step per candidate.

**Worked example (one metric, one candidate)**

| Per-step failure logits | Fail logit | P(fail) |
| --- | --- | --- |
| all 41 at 0 (untrained head) | log 41 = 3.71 | 0.976 |
| all 41 at −6 | −6 + 3.71 = −2.29 | 0.092 |
| 40 at −6, one at +2 | 2.01 | 0.882 |
| all 41 at −8.3 | −4.6 | 0.010 |

Two consequences follow:

- One confident failing step is enough to fail the plan, which is what we want.
- Safe steps add up. To report P(fail) = 1%, every step must sit near −8.3, not −4.6, because the 41 small risks are summed. Long plans need more confident per-step logits than short ones.

**Note on the code comment.** `__init__` says that with all step logits at 0, "P(pass) \~ 0.98". The math above gives P(fail) ≈ 0.98, so P(pass) ≈ 0.024. `step_head` has no special bias init, so an untrained model starts out predicting that almost every candidate fails these metrics. Most labels are passes, so early training mainly pushes all step logits down. That is probably harmless, but the comment is wrong. A bias init of about −8 on the four failure outputs would start the model near the label base rate.

### 3. Speed limit: mean compliance

The fifth head output gives a per-step compliance probability c\_t = σ(s\_t). The candidate's value is their mean, turned back into a logit for BCE:

```latex
\text{speed logit} = \operatorname{logit}\Big(\tfrac{1}{41}\textstyle\sum_t \sigma(s_t)\Big)
```

This copies nuPlan, which averages overspeed over the horizon, so a short burst of speeding costs only a little.

### 4. Comfort: attention pooling

Comfort checks acceleration and jerk bounds over the whole trajectory. It is not a per-step pass/fail, so it is read out by attention pooling, a third MHA. Its single query is a learned vector q ∈ ℝ^192 shared by all candidates, and its keys and values are the 41 step vectors:

```latex
\text{pooled} = \big[\mathrm{head}_1;\dots;\mathrm{head}_6\big] W^O,\qquad \mathrm{head}_i = \mathrm{softmax}\!\Big(\tfrac{q W_i^Q (X W_i^K)^{\top}}{\sqrt{32}}\Big)\, X W_i^V
```

Each head picks its own weights over the 41 steps. One head can concentrate on a jerk spike while another averages over the plan. `comfort_head` (192 → 1) then gives the comfort logit. The query does not depend on the candidate, so the pool asks the same question of every plan, and the answer differs only through X.

### 5. Progress: raw metres, gated and normalized

PDM's progress is relative: a candidate's distance along the route divided by the best safe candidate's. The model copies those steps:

1. Raw progress r\_k = softplus(progress\_head(x\_40)) × 10 m, read from the last state only. Softplus keeps it non-negative.
2. Gate by predicted safety: g\_k = r\_k · P(collision) · P(drivable) · P(direction). An unsafe plan's progress does not count.
3. Normalize by the best of the 16: progress\_k = g\_k / max\_j g\_j. If the best is under 0.1 m (PDM's threshold), progress\_k = the safety product instead.
4. Convert to a logit for the BCE.

Step 3 is the only place in w6's forward pass where candidates affect each other. Raw progress also gets its own loss (below), so the model learns metres directly as well as the normalized value.

### 6. Combined score and selection

The 7 sigmoid probabilities are combined with CLS's formula (`metrics.py`, `combined_score`):

```latex
S_k = P_{\text{col}} \cdot P_{\text{drive}} \cdot P_{\text{dir}} \cdot \frac{5\,P_{\text{prog}} + 5\,P_{\text{ttc}} + 4\,P_{\text{speed}} + 2\,P_{\text{comf}}}{16}
```

The verifier picks argmax\_k S\_k. This plugs probabilities into a formula defined on 0/1 metrics, which treats the metrics as independent. A small error in a multiplicative probability scales the whole score, which is one reason those three metrics dominate regret.

### 7. w6 training loss

| Term | Weight | What it trains |
| --- | --- | --- |
| BCE on all 7 logits vs. the cv labels | 1 | each metric's calibrated probability. Labels are soft: collision and direction can be 0.5. |
| Listwise ranking: cross-entropy between softmax\_k(S\_k / 0.02) and softmax\_k(S\*\_k / 0.02) | 3.0 | the order of the 16 candidates within one scene |
| Smooth L1 on raw progress / 10 m vs. the label's raw progress | 1.0 | metres directly |

The temperature 0.02 makes the target sharp: a score gap of 0.02 between two candidates is a factor of e ≈ 2.7 in target weight. The ranking term is the only part of the loss that compares candidates. BCE grades each candidate on its own, which is why it mostly learns scene-level risk.

### 8. w7: one score per candidate

w7 keeps the input, both decoder layers and the final LayerNorm. Then:

1. `score_pool`: the same attention pooling as comfort, with its own learned query and weights, gives one 192-dim vector per candidate.
2. `score_head` (192 → 1) gives a score s\_k.
3. Loss: cross-entropy between softmax\_k(s\_k) and a target over the 16. That is a one-hot on PDM+SL's best candidate (ties split evenly) with `--classify-target best`, or softmax(S\*/0.02) with the default `soft`. Scenes whose 16 candidates all tie are skipped, since they don't say which candidate is best.

|  | w6 (metrics) | w7 (classifier) |
| --- | --- | --- |
| Readout | step head + smooth max, mean, pool, progress gating | one attention pool |
| Supervision | 7 labels per candidate + ranking + raw metres | which candidate is best |
| Structure built in | PDM's any-step rule and score formula | none, so it must be learned |
| Information bottleneck | per-step logits, 41 per metric | one 192-dim vector per candidate |
| Result so far | regret 0.0352 | 0.0461 at epoch 3, seed 0 (still running) |

w7 gets a much weaker signal, one best index per scene against 7 labels per candidate. It also has to learn the "any bad step fails the plan" rule through a pool that averages by design. Its early regret is consistent with that. The comparison is final only after all 30 epochs and 3 seeds.

## Implications for our findings

Per-step cross-attention lets each future pose ask which scene tokens matter at that moment. That fits the w6 gains on collision and TTC. It cannot fix drivable area, because the keys do not hold the geometry that label depends on.

- **Why collision and TTC improved** (collision pair accuracy 0.57 → 0.78). In w5 one plan token had to summarize all 80 poses before it could attend to agents. In w6 the pose at t = 2.3 s gets its own query, so its attention can focus on the agents near that pose. Agent tokens hold only past tracks and the latest pose, though. Each query has to infer where an agent will be from its history.
- **Why drivable area is stuck** (within-scene pair accuracy 0.58 → 0.57). A lane is one key vector, placed by its midpoint. A dot product between a pose query and that key can tell "near this lane", but not "corner 0.2 m past the boundary". There are no car-park keys, and the ego key is constant, so ego size never enters attention.
- **Candidates are judged in isolation.** Since the 16 sequences never attend to each other, within-scene contrast comes only from the loss (the listwise ranking term, and w7's softmax over the 16). This matches the finding that BCE mostly rewards scene-level risk.

**Attention-side ideas** (untested)

- Richer keys: lane-boundary points or car-park polygons as extra memory tokens. Cross-attention cost grows linearly with the token count (41 × m per head).
- A geometric attention bias: add a learned function of the step-to-token distance to the logits (the B term in the softmax), so each pose looks first at nearby lanes and agents.
- A cross-candidate layer after the per-step layers: pool each candidate, then let the 16 pooled vectors self-attend, as w5 does.
- **Planned ablation: without self-attention.** Drop the step-to-step self-attention and keep cross-attention. This tests how much of w6's gain comes from temporal context and how much from per-step scene queries. It is listed under Planned ablations in the [Learned Verifier Architecture](https://claude.ai/artifact/XBsF7EvZaJx1p23ADsqebW) doc.
