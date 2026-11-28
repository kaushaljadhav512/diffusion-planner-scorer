"""
Verifier dataset over the label files of diffusion_planner.scorer.labels (task 2.10).

    train_files, val_files = split_by_scenario(find_samples(label_dir), val_frac=0.1)
    loader = DataLoader(LabelDataset(train_files), batch_size=64, shuffle=True)
"""
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion_planner.scorer.labels import load_sample

# Model inputs the verifier reads (same keys as DP's DataProcessor / sample_diversity.py)
INPUT_KEYS = [
    "ego_current_state", "neighbor_agents_past", "static_objects",
    "lanes", "lanes_speed_limit", "lanes_has_speed_limit",
    "route_lanes", "route_lanes_speed_limit", "route_lanes_has_speed_limit",
]


def find_samples(label_dir) -> List[Path]:
    """Every label sample under label_dir (one subdirectory per scenario), sorted."""
    return sorted(p for p in Path(label_dir).glob("*/*.npz") if not p.name.endswith(".tmp.npz"))


def split_by_scenario(files: Sequence[Path], val_frac: float = 0.1, seed: int = 0) -> Tuple[List[Path], List[Path]]:
    """
    Train / val split by scenario (the sample's directory), never by sample: samples of one scenario are a few steps
    apart and nearly identical, so a per-sample split would leak.

    Returns:
        train_files, val_files; at least one validation scenario when val_frac > 0
    """
    scenarios = sorted({Path(f).parent.name for f in files})
    random.Random(seed).shuffle(scenarios)
    n_val = max(1, round(val_frac * len(scenarios))) if val_frac > 0 else 0
    val = set(scenarios[:n_val])
    return [Path(f) for f in files if Path(f).parent.name not in val], [Path(f) for f in files if Path(f).parent.name in val]


def subsample_scenarios(files: Sequence[Path], frac: float, seed: int = 0) -> List[Path]:
    """
    The samples of a random fraction of the scenarios (learning curve). For one seed the subsets are nested: a
    smaller fraction's scenarios are a subset of a larger one's.

    Returns:
        files of round(frac * n_scenarios) scenarios, at least one
    """
    scenarios = sorted({Path(f).parent.name for f in files})
    random.Random(seed).shuffle(scenarios)
    keep = set(scenarios[:max(1, round(frac * len(scenarios)))])
    return [Path(f) for f in files if Path(f).parent.name in keep]


class LabelDataset(Dataset):
    """
    Items:
        inputs:          dict of normalized model inputs (INPUT_KEYS), as the planner feeds DP
        candidates:      [K, T, 4] ego plans, ego frame, unnormalized (the verifier normalizes them)
        metrics_oracle:  [K, 7] per-metric labels with logged futures
        metrics_cv:      [K, 7] the same with constant-velocity forecasts (PDM selection, as a baseline)
        progress_oracle: [K] raw progress along the centerline [m] behind metrics_oracle's normalized progress
        progress_cv:     [K] the same for metrics_cv
    """

    def __init__(self, files: Sequence[Path], sidecar: Optional[Path] = None):
        """sidecar: dir from scripts/make_epcv_sidecar.py (same layout as the labels); adds ego_speed [] (the
        simulated ego's speed, m/s) and progress_route [K] (computed progress along the route, m) to every item."""
        self.files = [Path(f) for f in files]
        self.sidecar = Path(sidecar).expanduser() if sidecar is not None else None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = load_sample(self.files[index])
        return {
            "inputs": {k: torch.from_numpy(sample["inputs"][k]) for k in INPUT_KEYS},
            "candidates": torch.from_numpy(sample["candidates"]).float(),
            "metrics_oracle": torch.from_numpy(sample["metrics_oracle"]).float(),
            "metrics_cv": torch.from_numpy(sample["metrics_cv"]).float(),
            "progress_oracle": torch.from_numpy(sample["progress_oracle"]).float(),
            "progress_cv": torch.from_numpy(sample["progress_cv"]).float(),
            **self._sidecar(self.files[index]),
        }

    def _sidecar(self, path: Path) -> Dict[str, torch.Tensor]:
        if self.sidecar is None:
            return {}
        z = np.load(self.sidecar / path.parent.name / path.name)
        return {"ego_speed": torch.tensor(float(z["ego_speed"])),
                "progress_route": torch.from_numpy(z["progress_route"]).float()}
