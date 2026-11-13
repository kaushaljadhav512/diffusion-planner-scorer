"""
Analytic (non-learned) versions of PDM's map-only metrics, computed from DP's model inputs alone. No nuPlan imports.

drivable_area_compliance re-implements PDM's drivable-area rule: a candidate fails if, at any step of the scored
horizon (current pose plus horizon poses), any of the 4 corners of ego's box is outside every drivable polygon.
The polygons are DP's lane / lane-connector inputs (centerline plus left and right boundary offsets), so compared with
PDM it lacks car parks, keeps only the lane_num lanes nearest to ego (20 points each), rebuilds lane-connector
polygons from their boundaries (nuPlan's own connector polygons can differ), and checks the plan's own poses rather
than PDM's LQR-tracked ones.
"""
from typing import Optional

import numpy as np

# nuPlan's get_pacifica_parameters()
HALF_LENGTH = 2.588
HALF_WIDTH = 1.1485
REAR_AXLE_TO_CENTER = 1.461

# normalization.json, "lanes": x is (x - 10) / 20, every other coordinate feature / 20
LANE_MEAN_X = 10.0
LANE_STD = 20.0


def lane_polygons(lanes: np.ndarray) -> np.ndarray:
    """
    Args:
        lanes: [L, N, 12] normalized DP lane inputs (x, y, dx, dy, left - center (2), right - center (2), tl (4));
               padded lanes are all zero

    Returns:
        polygons: [L', 2N, 2] ego-frame lane polygons (left boundary, then the right boundary reversed), padding dropped
    """
    lanes = np.asarray(lanes, dtype=np.float64)
    valid = np.abs(lanes).sum(axis=(-1, -2)) > 0
    lanes = lanes[valid]
    center = lanes[..., 0:2] * LANE_STD
    center[..., 0] += LANE_MEAN_X
    left = center + lanes[..., 4:6] * LANE_STD
    right = center + lanes[..., 6:8] * LANE_STD
    return np.concatenate([left, right[:, ::-1]], axis=1)


def points_in_polygons(points: np.ndarray, polygons: np.ndarray) -> np.ndarray:
    """
    Even-odd ray casting, vectorized over points and polygons.

    Args:
        points: [P, 2]
        polygons: [L, V, 2] closed implicitly (last vertex connects to the first)

    Returns:
        inside: [L, P] bool
    """
    a = polygons[:, None, :, :]                               # [L, 1, V, 2]
    b = np.roll(polygons, -1, axis=1)[:, None, :, :]
    x, y = points[None, :, None, 0], points[None, :, None, 1]  # [1, P, 1]
    ay, by = a[..., 1], b[..., 1]
    straddles = (ay > y) != (by > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = a[..., 0] + (y - ay) * (b[..., 0] - a[..., 0]) / (by - ay)
    crossings = straddles & (x < x_cross)
    return (crossings.sum(axis=-1) % 2) == 1


def ego_corners(plans: np.ndarray) -> np.ndarray:
    """
    Args:
        plans: [K, T, 4] rear-axle poses (x, y, cos, sin), ego frame

    Returns:
        corners: [K, T, 4, 2] front-left, front-right, rear-left, rear-right of ego's box
    """
    xy, cos, sin = plans[..., 0:2], plans[..., 2], plans[..., 3]
    norm = np.sqrt(cos ** 2 + sin ** 2)
    norm = np.where(norm > 0, norm, 1.0)
    cos, sin = cos / norm, sin / norm
    forward = np.stack([cos, sin], axis=-1)                  # [K, T, 2]
    left = np.stack([-sin, cos], axis=-1)
    center = xy + REAR_AXLE_TO_CENTER * forward
    offsets = [(HALF_LENGTH, HALF_WIDTH), (HALF_LENGTH, -HALF_WIDTH), (-HALF_LENGTH, HALF_WIDTH), (-HALF_LENGTH, -HALF_WIDTH)]
    return np.stack([center + lo * forward + la * left for lo, la in offsets], axis=-2)


def points_near_polygons(points: np.ndarray, polygons: np.ndarray, tolerance: float) -> np.ndarray:
    """
    Args:
        points: [P, 2]; polygons: [L, V, 2]; tolerance: [m]

    Returns:
        near: [L, P] bool, the point is inside the polygon or within tolerance of its boundary
    """
    inside = points_in_polygons(points, polygons)
    if tolerance <= 0:
        return inside
    a = polygons[:, None]                                     # [L, 1, V, 2]
    ab = np.roll(polygons, -1, axis=1)[:, None] - a
    p = points[None, :, None]                                 # [1, P, 1, 2]
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-12), 0.0, 1.0)
    dist = np.linalg.norm(a + t[..., None] * ab - p, axis=-1).min(-1)
    return inside | (dist <= tolerance)


def drivable_area_compliance(lanes: np.ndarray, plans: np.ndarray, horizon: int = 40,
                             include_current: bool = True, tolerance: float = 0.2,
                             polygons: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Args:
        lanes: [L, N, 12] normalized DP lane inputs of one scene
        plans: [K, T, 4] candidate ego plans, ego frame, unnormalized
        horizon: plan poses scored (PDM: 40 = 4 s)
        include_current: also check the current pose (PDM scores the current state too)
        tolerance: a corner within this distance [m] of a lane polygon counts as inside. Neighbouring lane polygons
                   rebuilt from 20-point boundaries leave thin slivers between them; 0.2 m closes them (on 300 val
                   scenes, false off-road went from 160 to 1 candidate)
        polygons: precomputed lane_polygons(lanes)

    Returns:
        scores: [K] 1.0 if every corner stays on the lanes at every scored step, else 0.0
    """
    distance = corner_lane_distance(lanes, plans, horizon, include_current, polygons)
    return (distance <= tolerance).all(axis=(1, 2)).astype(np.float64)


def corner_lane_distance(lanes: np.ndarray, plans: np.ndarray, horizon: int = 40, include_current: bool = True,
                         polygons: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Distance of every corner of ego's box to the nearest lane polygon, 0 inside one. Distances are only computed for
    the (few) corners outside every polygon, so one call serves every tolerance.

    Returns:
        distance: [K, T', 4] metres (T' = horizon, plus 1 with include_current); inf if the scene has no lanes
    """
    plans = np.asarray(plans, dtype=np.float64)[:, :horizon]
    if include_current:
        current = np.broadcast_to(np.array([0.0, 0.0, 1.0, 0.0]), (plans.shape[0], 1, 4))
        plans = np.concatenate([current, plans], axis=1)
    if polygons is None:
        polygons = lane_polygons(lanes)
    K, T = plans.shape[:2]
    if len(polygons) == 0:
        return np.full((K, T, 4), np.inf)
    corners = ego_corners(plans).reshape(-1, 2)
    distance = np.zeros(len(corners))
    outside = ~points_in_polygons(corners, polygons).any(axis=0)
    if outside.any():
        points = corners[outside]
        a = polygons[:, None]                                 # [L, 1, V, 2]
        ab = np.roll(polygons, -1, axis=1)[:, None] - a
        p = points[None, :, None]                             # [1, P, 1, 2]
        t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-12), 0.0, 1.0)
        distance[outside] = np.linalg.norm(a + t[..., None] * ab - p, axis=-1).min(axis=(0, 2))
    return distance.reshape(K, T, 4)
