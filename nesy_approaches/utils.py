"""Utility functions for training and dataset generation."""

import csv
import os
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from nesy_approaches.model1 import Model1Trainer
from nesy_data_gen.generators import (
    DatasetGeneratorFiltering,
    DatasetGeneratorFixGuided,
)


def seed_everything(seed: int):
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results_to_csv(metrics: dict, filename: str):
    """Save metrics to CSV file.

    Args:
        metrics: Dictionary of metric names to values.
        filename: Path to output CSV file.
    """
    file_exists = os.path.exists(filename)
    should_write_header = (not file_exists) or os.path.getsize(filename) == 0

    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if should_write_header:
            writer.writeheader()
        writer.writerow(metrics)


def generate_dataset(params, constraint, sup_regions=None, device="cpu"):
    """Generate dataset and split into train/valid/test sets.

    Args:
        params: Dictionary with dataset generation parameters.
        constraint: Constraint object for filtering.

    Returns:
        Tuple of (x_train, y_train, x_valid, y_valid, x_test, y_test) as tensors.
    """
    generator_type = params.get("generator_type", "fixing")
    generator_params = params.get("ds_params")

    if generator_type == "fixing":
        generator = DatasetGeneratorFixGuided(**generator_params)
    elif generator_type == "filtering":
        generator = DatasetGeneratorFiltering(**generator_params)
    else:
        raise ValueError(
            f"Unknown generator_type='{generator_type}'. "
            "Supported values are: 'filtering', 'fix_guided'."
        )

    result = generator.generate_dataset(constraint, seed=params["seed"], device=device)

    X = result["X"]
    Y = result["Y"]
    dataset_info = result.get("dataset_info", {})

    # Split data
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X,
        Y,
        test_size=params["test_size"],
        random_state=params["seed"],
    )
    valid_ratio = params["valid_size"] / (1 - params["test_size"])
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_temp, Y_temp, test_size=valid_ratio, random_state=params["seed"]
    )
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    Y_valid = torch.tensor(Y_valid, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    X_unsup = None

    if sup_regions is not None:
        regions_train = constraint.get_regions_for_points(X_train.numpy())
        regions_valid = constraint.get_regions_for_points(X_valid.numpy())
        sup_mask_train = torch.as_tensor(
            np.isin(regions_train, sup_regions), dtype=torch.bool
        )
        sup_mask_valid = torch.as_tensor(
            np.isin(regions_valid, sup_regions), dtype=torch.bool
        )

        X_unsup = torch.cat([X_train[~sup_mask_train], X_valid[~sup_mask_valid]])
        X_train = X_train[sup_mask_train]
        Y_train = Y_train[sup_mask_train]
        X_valid = X_valid[sup_mask_valid]
        Y_valid = Y_valid[sup_mask_valid]

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test, X_unsup, dataset_info


def generate_pseudo_labels(x_unsup, model, constraints=None, CL=None):
    """Generate pseudo-labels and report how much constraints moved them."""
    output_pseudo = model.inference(x_unsup).detach()
    stats = {"pseudo_label_shift_mean_distance": 0.0}

    if constraints is not None:
        output_np = output_pseudo.cpu().numpy()
        fixed_np = np.stack(constraints.fix(output_np, x_unsup.cpu().numpy()))
        abs_diff = np.sum(np.abs(fixed_np - output_np), axis=1)
        stats = {"pseudo_label_shift_mean_distance": float(np.mean(abs_diff))}
        y_unsup = torch.tensor(fixed_np, dtype=torch.float32)
    elif CL is not None:
        y_unsup = CL(output_pseudo)
        abs_diff = torch.sum(torch.abs(y_unsup - output_pseudo), dim=1)
        stats = {"pseudo_label_shift_mean_distance": float(abs_diff.mean().item())}
    else:
        y_unsup = output_pseudo.float()

    return y_unsup, stats


def add_unsup_data_proportionally(x_train, y_train, x_val, y_val, X_unsup, Y_unsup):
    """Add unsupervised data to train and val sets proportionally."""

    # Calculate original train/val ratio
    total_supervised = len(x_train) + len(x_val)
    val_ratio = len(x_val) / total_supervised

    # Split unsupervised data with same ratio
    X_unsup_train, X_unsup_val, Y_unsup_train, Y_unsup_val = train_test_split(
        X_unsup, Y_unsup, test_size=val_ratio, random_state=42
    )

    # Concatenate
    x_train_new = np.concatenate([x_train, X_unsup_train], axis=0)
    y_train_new = np.concatenate([y_train, Y_unsup_train], axis=0)
    x_val_new = np.concatenate([x_val, X_unsup_val], axis=0)
    y_val_new = np.concatenate([y_val, Y_unsup_val], axis=0)

    # Shuffle train set
    train_idx = np.random.permutation(len(x_train_new))
    x_train_new = x_train_new[train_idx]
    y_train_new = y_train_new[train_idx]

    return x_train_new, y_train_new, x_val_new, y_val_new


def train_pipeline(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    constraint,
    config,
    x_unsup=None,
    return_info=False,
):
    """Run the training pipeline and return predictions.

    Args:
        x_train, y_train: Training data.
        x_val, y_val: Validation data.
        x_test: Test inputs.
        constraint: Constraint object.
        config: Configuration dict with "model" key.
        x_unsup: Optional unsupervised inputs for pseudo-labeling.

    Returns:
        Model predictions on test set, and optional training diagnostics.
    """
    trainer = Model1Trainer(config["model"])
    trainer.train(x_train, y_train, x_val, y_val, constraint=constraint)

    train_info = {"pseudo_label_shift_mean_distance": float("nan")}

    # Self-training with pseudo labels
    if config.get("unsupervised", False) and x_unsup is not None:
        # Generate pseudo labels for unsupervised data
        y_unsup, train_info = generate_pseudo_labels(x_unsup, trainer, constraint)

        # Add unsup data to train/val proportionally
        x_train_new, y_train_new, x_val_new, y_val_new = add_unsup_data_proportionally(
            x_train.numpy(),
            y_train.numpy(),
            x_val.numpy(),
            y_val.numpy(),
            x_unsup.numpy(),
            y_unsup.numpy(),
        )

        # Convert back to tensors
        x_train_new = torch.tensor(x_train_new, dtype=torch.float32)
        y_train_new = torch.tensor(y_train_new, dtype=torch.float32)
        x_val_new = torch.tensor(x_val_new, dtype=torch.float32)
        y_val_new = torch.tensor(y_val_new, dtype=torch.float32)

        # Retrain with augmented data
        trainer.train(
            x_train_new, y_train_new, x_val_new, y_val_new, constraint=constraint
        )

    predicted = trainer.inference(x_test).detach()
    if return_info:
        return predicted, train_info
    return predicted


def train_pipeline_with_unsupervised_sl(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    constraint,
    config,
    x_unsup,
    sl_module,
    return_info=False,
):
    """Run the training pipeline and return predictions.

    Args:
        x_train, y_train: Training data.
        x_val, y_val: Validation data.
        x_test: Test inputs.
        constraint: Constraint object.
        config: Configuration dict with "model" key.
        x_unsup: Unsupervised inputs for semantic loss
        sl_module: pytorch module that computes constraint probability from the outputs

    Returns:
        Model predictions on test set, and optional training diagnostics.
    """

    trainer = Model1Trainer(config["model"])
    trainer.train_with_unsupervised_sl(
        x_train, y_train, x_val, y_val, x_unsup, sl_module, constraint=constraint
    )

    train_info = {"pseudo_label_shift_mean_distance": float("nan")}

    predicted = trainer.inference(x_test).detach()
    if return_info:
        return predicted, train_info
    return predicted
