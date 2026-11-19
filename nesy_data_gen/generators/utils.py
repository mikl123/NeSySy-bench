"""Shared utilities for dataset generators."""

import csv
import os
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, X, constr_indices):
        self.X = X
        self.constr_indices = constr_indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.constr_indices[idx]


from nesy_data_gen.utils.seed import seed_everything  # noqa: F401


class RandomGenerator(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_dim=64, num_hidden=2, device="cpu"):
        super().__init__()

        layers = [nn.Linear(n_inputs, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, n_outputs), nn.Sigmoid()]

        self.net = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x):
        return self.net(x)


class CallableGeneratorFunction(nn.Module):
    """Wraps a user-provided callable as a no-parameter ``nn.Module``."""

    def __init__(self, fn: Callable):
        super().__init__()
        self._fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._fn(x)
        if not isinstance(out, torch.Tensor):
            out = torch.as_tensor(out, dtype=x.dtype, device=x.device)
        return out


def write_exploration_points_csv(
    path: str, x_np: np.ndarray, y_pred_np: np.ndarray, region_idx: np.ndarray
) -> None:
    """Write all exploration inputs/outputs with region id."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    rows = []
    for i in range(len(x_np)):
        row = {"sample_idx": int(i), "region_index": int(region_idx[i])}
        for j in range(x_np.shape[1]):
            row[f"x{j}"] = float(x_np[i, j])
        for j in range(y_pred_np.shape[1]):
            row[f"y_pred{j}"] = float(y_pred_np[i, j])
        rows.append(row)

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
