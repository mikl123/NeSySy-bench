"""Base class for dataset generators."""

import copy
import json
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from nesy_data_gen.constraints.base import Constraint
from nesy_data_gen.generators.utils import (
    CallableGeneratorFunction,
    MyDataset,
    RandomGenerator,
    seed_everything,
    write_exploration_points_csv,
)
from nesy_data_gen.utils.metrics import compute_dataset_metrics


class NoSatisfyingSamplesError(Exception):
    """Raised when no samples satisfying the constraint could be found."""

    pass


class DatasetGenerator(ABC):
    """Abstract base class for all dataset generators.

    Subclasses implement the region-specific sample collection strategy.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    _oversample_factor: float = 1.0

    def _postprocess_predictions(
        self, predictions: np.ndarray, constraint: Constraint
    ) -> np.ndarray:
        if not constraint.is_real:
            return (predictions > 0.5).astype(np.float32)
        return predictions

    def _train_diversity_model(
        self,
        x: torch.Tensor,
        x_to_regions,
        constraint,
        device: str = "cpu",
    ) -> nn.Module:
        """Train a diversity network on pre-generated data."""
        from nesy_data_gen.generators.config import LossConfig
        from nesy_data_gen.generators.losses import CompositeLoss

        loss_configs = self.losses
        if loss_configs is None:
            loss_configs = [LossConfig(name="soft_repulsion", weight=1.0)]

        composite_loss = CompositeLoss.from_configs(loss_configs)

        x = x.to(device)
        x_to_regions = np.asarray(x_to_regions)
        n_points, n_inputs = x.shape
        dataset = MyDataset(x, x_to_regions)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = RandomGenerator(
            n_inputs,
            self.n_outputs,
            hidden_dim=self.hidden_dim,
            num_hidden=self.num_hidden,
            device=device,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, threshold=1e-4
        )
        best_loss = float("inf")
        best_model_state = None
        epochs_no_improve = 0
        early_stop_patience = 20
        min_delta = 1e-6
        warmup_epochs = composite_loss.warmup_epochs()

        loss_history = {}  # {loss_name: [epoch_mean, ...]}

        for epoch in range(self.epochs):
            epoch_losses = []
            epoch_details = {}
            for inp, _ in dataloader:
                optimizer.zero_grad()
                inp = inp.to(device)
                outputs = model(inp)
                loss, details = composite_loss.call_detailed(
                    outputs, inp, model=model, constraint=constraint, epoch=epoch
                )
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                for k, v in details.items():
                    epoch_details.setdefault(k, []).append(v)

            mean_loss = float(np.mean(epoch_losses))
            for k, vals in epoch_details.items():
                loss_history.setdefault(k, []).append(float(np.mean(vals)))
            scheduler.step(mean_loss)

            # Skip early stopping during warmup (before all losses are active)
            if epoch < warmup_epochs:
                best_model_state = copy.deepcopy(model.state_dict())
                continue

            if best_loss - mean_loss > min_delta:
                best_loss = mean_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                break

        n_epochs_trained = epoch + 1
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        model.eval()
        model._training_info = {
            "n_epochs": n_epochs_trained,
            "best_loss": best_loss,
            "final_loss": mean_loss,
            "final_losses": (
                {name: vals[-1] for name, vals in loss_history.items()}
                if loss_history
                else {}
            ),
            "loss_history": loss_history,
        }

        if self.exploration_points_csv_path is not None:
            x_np = x.detach().cpu().numpy()
            with torch.no_grad():
                y_pred_np = model(x).detach().cpu().numpy()
            write_exploration_points_csv(
                self.exploration_points_csv_path, x_np, y_pred_np, x_to_regions
            )

        return model

    def _predict_outputs(
        self, model, current_x: np.ndarray, constraint: Constraint, device: str
    ) -> np.ndarray:
        outputs = (
            model(torch.tensor(current_x, dtype=torch.float32, device=device))
            .detach()
            .cpu()
            .numpy()
        )
        return self._postprocess_predictions(outputs, constraint)

    def generate_dataset(
        self,
        constraint: Constraint,
        seed: int,
        device: str = "cpu",
        generator_function: Any = None,
    ) -> Any:
        """Generate the dataset using shared exploration logic.

        If ``generator_function`` is provided, it is used as the mapping
        directly and training is skipped.
        """
        seed_everything(seed=seed)
        if self.distant_supervision and constraint.is_real:
            raise ValueError(
                "distant_supervision=True is only supported for non-real (Boolean) "
                "constraints, but constraint.is_real is True."
            )
        x = torch.rand(self.exploration_set_size, self.n_inputs, dtype=torch.float32)
        x_to_regions = constraint.get_regions_for_points(x.numpy())
        if generator_function is not None:
            model = CallableGeneratorFunction(generator_function).to(device)
            model.eval()
        else:
            print("Training base model...")
            model = self._train_diversity_model(x, x_to_regions, constraint, device)
            print("Finished training base model...")

        dataset_info = defaultdict(list)
        dataset_info["base_model_training"] = json.dumps(
            getattr(model, "_training_info", {})
        )

        x_region_final = []
        y_region_final = []
        n_regions = constraint.c * constraint.b
        desired_n_samples_per_region = int(self.n_samples / n_regions)
        print("Collecting region samples...")
        for region_index in set(x_to_regions):

            x_region = constraint.sample_from_region(
                region_index,
                int(desired_n_samples_per_region * self._oversample_factor),
                self.n_inputs,
            )
            with torch.no_grad():
                y_region_pred = (
                    model(torch.tensor(x_region, dtype=torch.float32, device=device))
                    .detach()
                    .cpu()
                    .numpy()
                )
            y_region_pred = self._postprocess_predictions(y_region_pred, constraint)

            region_x, region_y = self._collect_region_samples(
                region_index=region_index,
                x_region=x_region,
                y_region_pred=y_region_pred,
                constraint=constraint,
                desired_n_samples_per_region=desired_n_samples_per_region,
                dataset_info=dataset_info,
                model=model,
                device=device,
            )
            if region_x is None or len(region_x) == 0:
                continue
            x_region_final.append(region_x)
            y_region_final.append(region_y)

        x_out = np.concatenate(x_region_final)
        y_out = np.concatenate(y_region_final)
        indices = np.random.permutation(len(x_out))
        x_out = x_out[indices]
        y_out = y_out[indices]

        dataset_info["n_samples_generated"] = len(x_out)

        if len(x_out) < self.n_samples:
            warnings.warn(
                f"Not enough samples found {len(x_out)} / {self.n_samples}. Consider increasing the exploration set size."
            )
            dataset_info["deficit_samples"] = True
        else:
            dataset_info["deficit_samples"] = True
        print("Verifying constraint satisfaction...")
        if constraint.is_real:
            assert all(
                constraint.check(y_out, x_out)
            ), "Some samples do not satisfy the constraint"
        else:
            assert np.isin(y_out, [0, 1]).all(), "Some samples are not binary"
            assert all(
                constraint.check(y_out, x_out)
            ), "Some samples do not satisfy the constraint"

        dataset_info.update(compute_dataset_metrics(x_out, y_out, constraint))

        regions_final = constraint.get_regions_for_points(x_out)
        dataset_info["per_region_metrics"] = {}
        for r in sorted(set(regions_final)):
            mask = regions_final == r
            dataset_info["per_region_metrics"][r] = compute_dataset_metrics(
                x_out[mask], y_out[mask], constraint
            )

        if self.distant_supervision:
            neg_x_list = dataset_info.pop("negative_x", [])
            neg_y_list = dataset_info.pop("negative_y", [])
            if len(neg_x_list) == 0:
                warnings.warn(
                    "distant_supervision is True but no negative samples were collected."
                )
            else:
                neg_x = np.stack(neg_x_list)
                neg_y = np.stack(neg_y_list)
                n_balanced = min(len(x_out), len(neg_x))
                # Downsample both to equal counts
                pos_idx = np.random.permutation(len(x_out))[:n_balanced]
                neg_idx = np.random.permutation(len(neg_x))[:n_balanced]
                x_pos, y_pos = x_out[pos_idx], y_out[pos_idx]
                x_neg, y_neg = neg_x[neg_idx], neg_y[neg_idx]
                # Append label column: 1.0 = satisfies constraint, 0.0 = violates
                y_pos = np.hstack([y_pos, np.ones((n_balanced, 1), dtype=np.float32)])
                y_neg = np.hstack([y_neg, np.zeros((n_balanced, 1), dtype=np.float32)])
                x_out = np.concatenate([x_pos, x_neg])
                y_out = np.concatenate([y_pos, y_neg])
                shuffle_idx = np.random.permutation(len(x_out))
                x_out = x_out[shuffle_idx]
                y_out = y_out[shuffle_idx]
                n_unique_pos = len(np.unique(y_pos, axis=0))
                n_unique_neg = len(np.unique(y_neg, axis=0))
                dataset_info["distant_supervision"] = {
                    "n_positive": n_balanced,
                    "n_negative": n_balanced,
                    "n_unique_positive": n_unique_pos,
                    "n_unique_negative": n_unique_neg,
                }

        return {
            "X": x_out,
            "Y": y_out,
            "constraint": constraint,
            "dataset_info": dataset_info,
        }

    @abstractmethod
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
        """Collect satisfying samples for a single region."""
        pass
