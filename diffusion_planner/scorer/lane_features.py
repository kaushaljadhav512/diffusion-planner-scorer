"""
Per-step local lane features for the timestep verifier (Verifier(step_lane=...), train.py --step-lane).

Every scored state of every candidate (the current state plus the HORIZON plan poses) gets features of the lanes
around ego's box at that state. They are computed without gradient from DP's raw lane inputs (20 points per lane:
centerline plus left and right boundary offsets), in the state's own frame: origin at the centre of ego's box, x along
its heading. The model adds them to the state's token, so the decoder's self-attention shares them across states.

scalars [B, K, T, STEP_LANE_SCALARS] ("scalars" and "segments")
    0-3   signed distance of each corner of ego's box to the lanes [m / 2, clipped to +-2]. Outside every lane polygon:
          the distance to the nearest one (exact). Inside: minus the largest depth inside a polygon that holds the
          corner (a lower bound on the depth inside the union: a corner near a seam between two lanes reads shallow)
    4     lane polygons holding at least one corner / 3
    5     ego's centre inside a route-lane polygon (0 / 1)
    6     signed distance of the centre to the route-lane polygons [m / 5, clipped to +-2], as 0-3
    7-8   cos, sin of ego's heading relative to the lane under ego (the nearest centerline segment's direction)
    9-10  that lane's speed limit (as DP's input: m/s / 20), has-speed-limit
    11    red light on that lane
    12    lateral offset of the centre from the middle of that lane [m / 2], positive to the left
    13    that lane's width at the centre (distance to its left plus its right boundary) [m / 4]
    "The lane under ego": among the polygons holding the centre, the one best aligned with ego's heading; if none
    holds it, the nearest polygon.

segments [B, K, T, M, SEGMENT_FEATURES], mask [B, K, T, M] ("segments" only; True = empty): the n_boundary boundary
segments and the n_center centerline segments of the input lanes nearest to ego's centre (a segment joins two
consecutive points of a polyline), empty beyond radius
    0-3   endpoints in the state's frame [m / 10, clipped to +-3]
    4-6   one-hot: left boundary, right boundary, centerline
    7-10  signed distance of each corner of ego's box to the segment [m / 5, clipped to +-2], positive on the lane's
          side (right of a left boundary, left of a right boundary, left of a centerline)
    11-12 direction of the segment in the state's frame (cos, sin)
    13-14 its lane's speed limit (normalized as DP's input), has-speed-limit
    15-18 its lane's traffic light one-hot (green, yellow, red, unknown)
    19    its lane is on the route (its centerline matches one of the route lanes)
"""
from typing import Dict

import torch

from diffusion_planner.scorer.analytic_metrics import HALF_LENGTH, HALF_WIDTH, LANE_MEAN_X, LANE_STD, \
    REAR_AXLE_TO_CENTER

STEP_LANE_MODES = ("none", "scalars", "segments")
STEP_LANE_SCALARS = 14
SEGMENT_FEATURES = 20
RED = 2                     # index of red in the traffic light one-hot (nuPlan LaneSegmentTrafficLightData)
ROUTE_MATCH = 0.5           # [m] a lane is on the route if its centerline is this close (mean over points) to a route lane
# Corners of ego's box in its own frame (origin at the box centre): front-left, front-right, rear-left, rear-right
CORNERS = ((HALF_LENGTH, HALF_WIDTH), (HALF_LENGTH, -HALF_WIDTH), (-HALF_LENGTH, HALF_WIDTH), (-HALF_LENGTH, -HALF_WIDTH))


def scored_states(candidates: torch.Tensor, horizon: int) -> torch.Tensor:
    """[B, K, T, 4] plans -> [B, K, horizon + 1, 4]: the current state (origin, heading +x) and the first horizon poses."""
    plans = candidates[:, :, :horizon].float()
    current = plans.new_tensor([0.0, 0.0, 1.0, 0.0]).expand(*plans.shape[:2], 1, 4)
    return torch.cat([current, plans], dim=2)


def polylines(lanes: torch.Tensor):
    """Normalized DP lane inputs [B, L, N, 12] -> centre, left, right [B, L, N, 2] in metres (ego frame), valid [B, L]."""
    lanes = lanes.float()
    valid = lanes[..., :8].abs().sum(dim=(-1, -2)) > 0
    center = lanes[..., 0:2] * LANE_STD + lanes.new_tensor([LANE_MEAN_X, 0.0])
    return center, center + lanes[..., 4:6] * LANE_STD, center + lanes[..., 6:8] * LANE_STD, valid


def _segment_distance(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """
    Distance of points to segments, broadcast: p [..., 2], a / b [..., 2] segment ends.

    Returns:
        distance [...], cross [...] (> 0 when p is left of the direction a -> b)
    """
    ab, ap = b - a, p - a
    t = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0.0, 1.0)
    distance = torch.linalg.norm(ap - t[..., None] * ab, dim=-1)
    return distance, ab[..., 0] * ap[..., 1] - ab[..., 1] * ap[..., 0]


def _inside_and_edge(points: torch.Tensor, polygons: torch.Tensor, valid: torch.Tensor):
    """
    points [b, P, 2], polygons [b, L, V, 2] (closed implicitly), valid [b, L] ->
    inside [b, L, P] (even-odd ray casting, False for padding), edge [b, L, P] distance to the polygon's boundary (inf
    for padding).
    """
    a = polygons[:, :, None]                                       # [b, L, 1, V, 2]
    nxt = polygons.roll(-1, dims=2)[:, :, None]
    x, y = points[:, None, :, None, 0], points[:, None, :, None, 1]  # [b, 1, P, 1]
    ay, by = a[..., 1], nxt[..., 1]
    straddles = (ay > y) != (by > y)
    x_cross = a[..., 0] + (y - ay) * (nxt[..., 0] - a[..., 0]) / torch.where(by == ay, torch.ones_like(by), by - ay)
    inside = ((straddles & (x < x_cross)).sum(-1) % 2 == 1) & valid[:, :, None]
    edge = _segment_distance(points[:, None, :, None], a, nxt)[0].amin(-1)
    edge = torch.where(valid[:, :, None], edge, torch.full_like(edge, float("inf")))
    return inside, edge


def _signed_to_union(inside: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
    """[b, L, P] inside / edge distance -> [b, P] signed distance to the union of polygons: outside, the distance to the
    nearest polygon; inside, minus the largest depth inside a polygon holding the point."""
    depth = torch.where(inside, edge, torch.zeros_like(edge)).amax(1)
    return torch.where(inside.any(1), -depth, edge.amin(1))


def on_route(lanes: torch.Tensor, route_lanes: torch.Tensor) -> torch.Tensor:
    """[B, L] bool: the lane's centerline is within ROUTE_MATCH metres (mean over its points) of a route lane's."""
    center, _, _, valid = polylines(lanes)
    route, _, _, route_valid = polylines(route_lanes)
    gap = torch.linalg.norm(center[:, :, None] - route[:, None], dim=-1).mean(-1)   # [B, L, R]
    gap = torch.where(route_valid[:, None], gap, torch.full_like(gap, float("inf")))
    return (gap.amin(-1) < ROUTE_MATCH) & valid


@torch.no_grad()
def step_lane_features(inputs: Dict[str, torch.Tensor], candidates: torch.Tensor, horizon: int = 40,
                       segments: bool = False, n_boundary: int = 32, n_center: int = 4, radius: float = 10.0,
                       chunk: int = 4) -> Dict[str, torch.Tensor]:
    """
    Args:
        inputs: normalized DP inputs with lanes [B, L, N, 12], lanes_speed_limit [B, L, 1], lanes_has_speed_limit
                [B, L, 1], route_lanes [B, R, N, 12]
        candidates: [B, K, T, 4] plans (x, y, cos, sin), ego frame, unnormalized, T >= horizon
        segments: also return the nearest segments (see the module docstring)
        n_boundary, n_center, radius: segments kept per state, and the distance [m] beyond which they are empty
        chunk: samples processed at once (the largest intermediates are [chunk, L, K * T * 4, 2N])

    Returns:
        {"scalars": [B, K, T', STEP_LANE_SCALARS]} (T' = horizon + 1), plus "segments" [B, K, T', M, SEGMENT_FEATURES]
        and "segment_mask" [B, K, T', M] with segments
    """
    states = scored_states(candidates, horizon)                    # [B, K, T', 4]
    B, K, T = states.shape[:3]
    heading = states[..., 2:4] / torch.linalg.norm(states[..., 2:4], dim=-1, keepdim=True).clamp_min(1e-6)
    fwd = heading.reshape(B, K * T, 2)
    lft = torch.stack([-fwd[..., 1], fwd[..., 0]], dim=-1)
    centre = states[..., :2].reshape(B, K * T, 2) + REAR_AXLE_TO_CENTER * fwd   # [B, P, 2]
    local = centre.new_tensor(CORNERS)                                          # [4, 2] in the state's frame
    corners = centre[:, :, None] + local[:, 0, None] * fwd[:, :, None] + local[:, 1, None] * lft[:, :, None]

    center_l, left_l, right_l, valid = polylines(inputs["lanes"])               # [B, L, N, 2], [B, L]
    polygons = torch.cat([left_l, right_l.flip(-2)], dim=-2)                    # [B, L, 2N, 2]
    _, route_left, route_right, route_valid = polylines(inputs["route_lanes"])
    route_polygons = torch.cat([route_left, route_right.flip(-2)], dim=-2)
    speed = inputs["lanes_speed_limit"].float()[..., 0]                         # [B, L]
    has_speed = inputs["lanes_has_speed_limit"].float()[..., 0]
    light = inputs["lanes"].float()[:, :, 0, 8:12]                              # [B, L, 4]
    routed = on_route(inputs["lanes"], inputs["route_lanes"]).float()           # [B, L]
    L, S = center_l.shape[1], center_l.shape[2] - 1

    scalars = centre.new_zeros(B, K * T, STEP_LANE_SCALARS)
    if segments:
        M = n_boundary + n_center
        seg_out = centre.new_zeros(B, K * T, M, SEGMENT_FEATURES)
        seg_mask = torch.ones(B, K * T, M, dtype=torch.bool, device=centre.device)

    for s in range(0, B, chunk):
        sl = slice(s, s + chunk)
        b = centre[sl].shape[0]
        P = K * T
        c, f, l, ok = centre[sl], fwd[sl], lft[sl], valid[sl]

        # Corners against the lane polygons
        inside, edge = _inside_and_edge(corners[sl].reshape(b, P * 4, 2), polygons[sl], ok)
        signed = _signed_to_union(inside, edge).view(b, P, 4)
        scalars[sl, :, 0:4] = (signed / 2.0).clamp(-2.0, 2.0)
        scalars[sl, :, 4] = inside.view(b, L, P, 4).any(-1).sum(1).float() / 3.0

        # Centre against the route-lane polygons
        r_inside, r_edge = _inside_and_edge(c, route_polygons[sl], route_valid[sl])
        scalars[sl, :, 5] = r_inside.any(1).float()
        scalars[sl, :, 6] = (_signed_to_union(r_inside, r_edge) / 5.0).clamp(-2.0, 2.0)

        # Centre against each lane: its polygon, both boundaries, its centerline
        c_inside, c_edge = _inside_and_edge(c, polygons[sl], ok)                 # [b, L, P]
        p = c[:, None, :, None]                                                  # [b, 1, P, 1, 2]
        dists = {}
        for name, line in (("left", left_l[sl]), ("right", right_l[sl]), ("center", center_l[sl])):
            a, e = line[:, :, None, :-1], line[:, :, None, 1:]                   # [b, L, 1, S, 2]
            d, _ = _segment_distance(p, a, e)                                    # [b, L, P, S]
            dists[name] = torch.where(ok[:, :, None, None], d, torch.full_like(d, float("inf")))
        d_left, d_right = dists["left"].amin(-1), dists["right"].amin(-1)       # [b, L, P]
        nearest = dists["center"].argmin(-1)                                     # [b, L, P]
        tangent = center_l[sl, :, 1:] - center_l[sl, :, :-1]                     # [b, L, S, 2]
        tangent = tangent / torch.linalg.norm(tangent, dim=-1, keepdim=True).clamp_min(1e-6)
        tangent = tangent.gather(2, nearest[..., None].expand(-1, -1, -1, 2))   # [b, L, P, 2]
        cos = (tangent * f[:, None]).sum(-1)
        sin = tangent[..., 0] * f[:, None, :, 1] - tangent[..., 1] * f[:, None, :, 0]

        # The lane under ego: best aligned among those holding the centre, else the nearest
        key = torch.where(c_inside, 10.0 + cos, -c_edge)
        key = torch.where(ok[:, :, None], key, torch.full_like(key, -float("inf")))
        j = key.argmax(1, keepdim=True)                                          # [b, 1, P]
        pick = lambda t: t.gather(1, j)[:, 0]                                    # [b, L, P] -> [b, P]
        per_lane = lambda v: v[:, :, None].expand(-1, -1, P)                     # [b, L] -> [b, L, P]
        any_lane = ok.any(1, keepdim=True).float()                               # [b, 1]
        dl, dr = pick(d_left), pick(d_right)
        finite = torch.isfinite(dl) & torch.isfinite(dr)
        dl, dr = torch.where(finite, dl, torch.zeros_like(dl)), torch.where(finite, dr, torch.zeros_like(dr))
        scalars[sl, :, 7] = pick(cos) * any_lane
        scalars[sl, :, 8] = pick(sin) * any_lane
        scalars[sl, :, 9] = pick(per_lane(speed[sl])) * any_lane
        scalars[sl, :, 10] = pick(per_lane(has_speed[sl])) * any_lane
        scalars[sl, :, 11] = pick(per_lane(light[sl, :, RED])) * any_lane
        scalars[sl, :, 12] = ((dr - dl) / 2.0 / 2.0).clamp(-2.0, 2.0)
        scalars[sl, :, 13] = ((dl + dr) / 4.0).clamp(0.0, 3.0)

        if not segments:
            continue
        # Nearest segments to the centre: boundaries (left then right, lane-major) and centerlines
        bound_d = torch.cat([dists["left"], dists["right"]], dim=-1)             # [b, L, P, 2S]
        bound_d = bound_d.permute(0, 2, 1, 3).reshape(b, P, L * 2 * S)
        center_d = dists["center"].permute(0, 2, 1, 3).reshape(b, P, L * S)
        bd, bi = bound_d.topk(min(n_boundary, L * 2 * S), dim=-1, largest=False)
        cd, ci = center_d.topk(min(n_center, L * S), dim=-1, largest=False)

        def ends(lines):                                                         # [b, L, N, 2] x n -> [b, L * n * S, 2] x 2
            a = torch.cat([x[:, :, :-1] for x in lines], dim=2).reshape(b, -1, 2)
            e = torch.cat([x[:, :, 1:] for x in lines], dim=2).reshape(b, -1, 2)
            return a, e

        groups = []
        ba, be = ends([left_l[sl], right_l[sl]])
        groups.append((bi, bd, ba, be, 2 * S, lambda idx: (idx % (2 * S)) // S))  # side 0 left, 1 right
        ca, ce = ends([center_l[sl]])
        groups.append((ci, cd, ca, ce, S, lambda idx: torch.full_like(idx, 2)))
        start = 0
        for idx, dist, a_all, e_all, per, side_of in groups:
            n = idx.shape[-1]
            flat = idx.reshape(b, -1)
            a = a_all.gather(1, flat[..., None].expand(-1, -1, 2)).view(b, P, n, 2)
            e = e_all.gather(1, flat[..., None].expand(-1, -1, 2)).view(b, P, n, 2)
            lane = idx // per                                                    # [b, P, n]
            side = side_of(idx)
            # To the state's frame
            to_local = lambda q: torch.stack([((q - c[:, :, None]) * f[:, :, None]).sum(-1),
                                              ((q - c[:, :, None]) * l[:, :, None]).sum(-1)], dim=-1)
            a, e = to_local(a), to_local(e)                                      # [b, P, n, 2]
            d, cross = _segment_distance(local[None, None, None], a[..., None, :], e[..., None, :])  # [b, P, n, 4]
            sign = torch.where(side == 0, -1.0, 1.0)[..., None]                  # lane side: right of a left boundary
            corner_d = torch.where(cross * sign >= 0, d, -d)
            direction = (e - a) / torch.linalg.norm(e - a, dim=-1, keepdim=True).clamp_min(1e-6)
            take = lambda v: v.gather(1, lane.reshape(b, -1)).view(b, P, n)     # [b, L] -> [b, P, n]
            feats = torch.cat([
                (torch.cat([a, e], dim=-1) / 10.0).clamp(-3.0, 3.0),
                torch.nn.functional.one_hot(side, 3).float(),
                (corner_d / 5.0).clamp(-2.0, 2.0),
                direction,
                take(speed[sl])[..., None], take(has_speed[sl])[..., None],
                torch.stack([take(light[sl, :, i]) for i in range(4)], dim=-1),
                take(routed[sl])[..., None],
            ], dim=-1)
            empty = ~(dist <= radius)                                            # inf (padding) or too far
            seg_out[sl, :, start:start + n] = torch.where(empty[..., None], torch.zeros_like(feats), feats)
            seg_mask[sl, :, start:start + n] = empty
            start += n

    out = {"scalars": scalars.view(B, K, T, STEP_LANE_SCALARS)}
    if segments:
        out["segments"] = seg_out.view(B, K, T, M, SEGMENT_FEATURES)
        out["segment_mask"] = seg_mask.view(B, K, T, M)
    return out
