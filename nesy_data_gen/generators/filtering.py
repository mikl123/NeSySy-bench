import numpy as np

from nesy_data_gen.constraints.base import Constraint
from nesy_data_gen.generators.base import DatasetGenerator, NoSatisfyingSamplesError
from nesy_data_gen.generators.config import DatasetGeneratorConfig


class DatasetGeneratorFiltering(DatasetGenerator, DatasetGeneratorConfig):
    """Dataset generator that filters samples to satisfy constraints.

    Args: See DatasetGeneratorConfig for available parameters.
    """

    _oversample_factor: float = 1.5

    @staticmethod
    def sample_near(vectors, n, noise_std):
        """
        vectors: (M, D)
        n: number of new samples
        """
        M, D = vectors.shape
        idx = np.random.randint(0, M, (n,))
        base = vectors[idx]
        noise = noise_std * np.random.randn(n, D)
        samples = base + noise
        samples = np.clip(samples, 0.0, 1.0)
        return samples

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
    ) -> tuple[np.ndarray, np.ndarray]:
        saved_outputs = []
        saved_x = []
        current_x = x_region

        for iteration in range(1 + self.max_iterations):
            if len(saved_outputs) >= desired_n_samples_per_region:
                break

            if iteration > 0:
                current_x = self.sample_near(
                    np.array(saved_x),
                    desired_n_samples_per_region * 3,
                    noise_std=self.noise_std,
                )
                region_mask = (
                    constraint.get_regions_for_points(current_x) == region_index
                )
                current_x = current_x[region_mask]

            outputs_np = self._predict_outputs(model, current_x, constraint, device)
            satisfying_mask = np.array(constraint.check(outputs_np, current_x))

            if iteration == 0:
                if np.sum(satisfying_mask) == 0:
                    raise NoSatisfyingSamplesError(
                        "No satisfying samples found. Consider increasing the exploration set size. "
                        "Or using alternative dataset generator if dimensionality is high."
                    )
                dataset_info["first_valid_set_size"].append(
                    (region_index, np.sum(satisfying_mask))
                )

            saved_outputs.extend(outputs_np[satisfying_mask])
            saved_x.extend(current_x[satisfying_mask])

            if self.distant_supervision:
                dataset_info["negative_x"].extend(current_x[~satisfying_mask])
                dataset_info["negative_y"].extend(outputs_np[~satisfying_mask])

        saved_x = np.stack(saved_x)
        saved_outputs = np.stack(saved_outputs)
        indices = np.random.permutation(len(saved_x))
        dataset_info["n_samples_per_region"].append((region_index, len(indices)))
        saved_x = saved_x[indices][:desired_n_samples_per_region]
        saved_outputs = saved_outputs[indices][:desired_n_samples_per_region]
        return saved_x, saved_outputs
