"""Unified dataset-level metrics."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nesy_data_gen.constraints.base import Constraint

REAL_DEAD_THRESHOLD = 0.1


def _percentiles(
    values: np.ndarray,
    prefix: str,
    quantiles: tuple[int, ...] = (25, 50, 75),
) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for q in quantiles:
        pcts = np.percentile(values, q, axis=0)
        for dim, val in enumerate(pcts):
            rows.append((f"{q}%", f"{prefix}{dim}", round(float(val), 4)))
    return rows


def _dead_variables(y: np.ndarray, is_real: bool) -> list[int]:
    """Output dims that carry no useful information."""
    dead = []
    for d in range(y.shape[1]):
        col = y[:, d]
        if is_real:
            if col.max() - col.min() < REAL_DEAD_THRESHOLD:
                dead.append(int(d))
        else:
            if np.all(col == col[0]):
                dead.append(int(d))
    return dead


def _class_distribution(y: np.ndarray) -> list[tuple[tuple[int, ...], int]]:
    """Count occurrences of each unique output vector, sorted descending."""
    counts = Counter(tuple(int(v) for v in row) for row in y)
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)


def _duplicate_count(x: np.ndarray, y: np.ndarray) -> int:
    combined = np.concatenate([x, y], axis=1)
    _, counts = np.unique(combined, axis=0, return_counts=True)
    return int(np.sum(counts[counts > 1]))


def _samples_per_region(regions: np.ndarray) -> list[tuple[int, int]]:
    unique, counts = np.unique(regions, return_counts=True)
    return [(int(r), int(c)) for r, c in zip(unique, counts)]


def compute_dataset_metrics(
    x: np.ndarray,
    y: np.ndarray,
    constraint: Constraint,
) -> dict:
    """Compute dataset-level metrics after generation.

    Returns a flat dict that can be merged into ``dataset_info``.
    """
    regions = constraint.get_regions_for_points(x)
    n = len(x)
    m: dict = {}

    m["n_samples"] = n
    m["samples_per_region"] = _samples_per_region(regions)
    m["input_percentiles"] = _percentiles(x, "x")
    m["dead_variables"] = _dead_variables(y, constraint.is_real)
    dup = _duplicate_count(x, y)
    m["duplicate_count"] = dup
    m["high_duplicate_ratio"] = (dup / n > 0.1) if n else False

    if constraint.is_real:
        m["output_percentiles"] = _percentiles(y, "y")
        m["output_range"] = [
            (int(d), round(float(y[:, d].min()), 4), round(float(y[:, d].max()), 4))
            for d in range(y.shape[1])
        ]
        m["per_dim_mean"] = [
            (int(d), round(float(y[:, d].mean()), 4)) for d in range(y.shape[1])
        ]
        m["per_dim_std"] = [
            (int(d), round(float(y[:, d].std()), 4)) for d in range(y.shape[1])
        ]
        m["n_low_std"] = sum(1 for _, s in m["per_dim_std"] if s < 0.05)
        m["low_quality"] = m["n_low_std"] == y.shape[1]
    else:
        n_unique = len(np.unique(y, axis=0))
        class_dist = _class_distribution(y)
        m["n_unique_outputs"] = n_unique
        m["output_class_distribution"] = class_dist
        probs = np.array([c for _, c in class_dist], dtype=np.float64) / n
        m["output_entropy"] = round(float(-np.sum(probs * np.log2(probs))), 4)
        m["per_dim_balance"] = [
            (int(d), round(float(y[:, d].mean()), 4)) for d in range(y.shape[1])
        ]
        m["class_collapse"] = n_unique < max(3, n * 0.01)
        m["heavy_skew"] = (class_dist[0][1] / n > 0.5) if class_dist else False
        m["low_quality"] = m["heavy_skew"]

        # Cardinality: average number of active labels (=1) per sample
        m["cardinality"] = round(float(y.sum(axis=1).mean()), 4)

    return m
