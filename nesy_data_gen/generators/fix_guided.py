"""Guaranteed dataset generator: diversity network → fix → sample near & filter → fallback."""

from typing import Any

import numpy as np

from nesy_data_gen.constraints.base import Constraint
from nesy_data_gen.generators.base import DatasetGenerator
from nesy_data_gen.generators.config import FixGuidedGeneratorConfig


class DatasetGeneratorFixGuided(DatasetGenerator, FixGuidedGeneratorConfig):
    """Guaranteed generator using fix-guided regional collection."""

    def _collect_region_samples(
        self,
        region_index: int,
        x_region: np.ndarray,
        y_region_pred: np.ndarray,
        constraint: Constraint,
        desired_n_samples_per_region: int,
        dataset_info,
        model,
        device: str,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        y_region_fixed = np.stack(constraint.fix(y_region_pred, x_region))
        already_valid = np.array(constraint.check(y_region_pred, x_region))

        fix_distances = np.abs(y_region_fixed - y_region_pred).sum(axis=1)
        dataset_info["fix_distance_mean"].append(
            (region_index, round(float(fix_distances.mean()), 4))
        )

        collected_x = list[Any](x_region[already_valid])
        collected_y = list(y_region_fixed[already_valid])
        dataset_info["first_valid_set_size"].append((region_index, len(collected_x)))
        noisy_valid_count = 0

        y_boundary = y_region_fixed[~already_valid]
        x_boundary = x_region[~already_valid]

        if constraint.is_real:
            noise_std = self.fix_noise_std
            for _ in range(self.fix_iterations):
                if (
                    len(collected_y) >= desired_n_samples_per_region
                    or len(y_boundary) == 0
                ):
                    break
                y_cand = np.clip(
                    y_boundary
                    + noise_std * np.random.randn(y_boundary.shape[0], self.n_outputs),
                    0.0,
                    1.0,
                ).astype(np.float32)
                valid = np.array(constraint.check(y_cand, x_boundary))
                if np.any(valid):
                    remaining = desired_n_samples_per_region - len(collected_y)
                    valid_indices = np.flatnonzero(valid)
                    take_count = min(remaining, len(valid_indices))
                    selected_indices = valid_indices[:take_count]
                    collected_x.extend(x_boundary[selected_indices])
                    collected_y.extend(y_cand[selected_indices])
                    noisy_valid_count += take_count
                    if len(collected_y) >= desired_n_samples_per_region:
                        break
                y_boundary = y_boundary[~valid]
                x_boundary = x_boundary[~valid]
        dataset_info["n_noisy_valid"].append((region_index, noisy_valid_count))

        remaining = max(0, desired_n_samples_per_region - len(collected_y))
        boundary_fallback_count = min(remaining, len(x_boundary))
        dataset_info["n_on_boundary"].append((region_index, boundary_fallback_count))
        if boundary_fallback_count > 0:
            collected_x.extend(x_boundary[:boundary_fallback_count])
            collected_y.extend(y_boundary[:boundary_fallback_count])

        if len(collected_x) == 0:
            return None, None

        saved_x = np.stack(collected_x)
        saved_y = np.stack(collected_y)
        indices = np.random.permutation(len(saved_x))
        saved_x = saved_x[indices][:desired_n_samples_per_region]
        saved_y = saved_y[indices][:desired_n_samples_per_region]
        dataset_info["n_samples_per_region"].append((region_index, len(saved_x)))
        return saved_x, saved_y
