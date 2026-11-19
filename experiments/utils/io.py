"""Experiment I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from nesy_data_gen.constraints.base import Constraint


def save_run(
    result_dir: Path,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_valid: torch.Tensor,
    y_valid: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    constraint,
) -> None:
    """Persist dataset splits and constraint state for a single experiment run."""
    result_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        result_dir / "dataset.npz",
        x_train=x_train.numpy(),
        y_train=y_train.numpy(),
        x_valid=x_valid.numpy(),
        y_valid=y_valid.numpy(),
        x_test=x_test.numpy(),
        y_test=y_test.numpy(),
    )
    constraint.save_state(str(result_dir / "constraint"))


def save_dataset(
    result_dir: Path,
    x_train,
    y_train,
    x_valid,
    y_valid,
    x_test,
    y_test,
    constraint,
    constr_params,
    ds_params,
    dataset_info,
) -> None:
    """Save dataset, constraint, and metadata to disk."""
    save_run(result_dir, x_train, y_train, x_valid, y_valid, x_test, y_test, constraint)
    np.save(result_dir / "constr_params.npy", constr_params, allow_pickle=True)
    np.save(result_dir / "ds_params.npy", ds_params, allow_pickle=True)
    np.save(result_dir / "dataset_info.npy", dataset_info, allow_pickle=True)


def load_dataset(result_dir: Path):
    """Load dataset, constraint, and metadata from disk.

    Returns (x_train, y_train, x_valid, y_valid, x_test, y_test,
             constraint, constr_params, ds_params, dataset_info).
    """
    data = np.load(result_dir / "dataset.npz")
    x_train = torch.tensor(data["x_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.float32)
    x_valid = torch.tensor(data["x_valid"], dtype=torch.float32)
    y_valid = torch.tensor(data["y_valid"], dtype=torch.float32)
    x_test = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.float32)
    constraint = Constraint.load_state(str(result_dir / "constraint"))
    constr_params = np.load(result_dir / "constr_params.npy", allow_pickle=True).item()
    ds_params = np.load(result_dir / "ds_params.npy", allow_pickle=True).item()
    dataset_info = np.load(result_dir / "dataset_info.npy", allow_pickle=True).item()
    return (
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
        constraint,
        constr_params,
        ds_params,
        dataset_info,
    )
