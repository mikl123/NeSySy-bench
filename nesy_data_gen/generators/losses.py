"""Pretraining loss functions for dataset generators.

Users can combine multiple losses with weights via LossConfig in the generator config.
New losses can be added by subclassing PretrainingLoss and registering in LOSS_REGISTRY.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class PretrainingLoss(ABC):
    """Base class for all pretraining losses."""

    @abstractmethod
    def __call__(
        self, outputs: torch.Tensor, inputs: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor: ...


class SoftRepulsionLoss(PretrainingLoss):
    """Soft repulsion: 1/(dist + eps) so all pairs contribute, closer pairs more."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(
        self, outputs: torch.Tensor, inputs: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        if outputs.shape[0] <= 1:
            return torch.tensor(0.0, device=outputs.device)

        dist = torch.cdist(outputs, outputs, p=2)
        mask = ~torch.eye(dist.size(0), device=dist.device).bool()
        return torch.mean(1.0 / (dist[mask] + self.eps))


class DecorrelationLoss(PretrainingLoss):
    """Penalize linear dependence between output dimensions.

    Minimizes the squared off-diagonal entries of the empirical
    covariance matrix, pushing output dimensions toward independence.
    """

    def __call__(
        self, outputs: torch.Tensor, inputs: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        if outputs.shape[0] <= 2 or outputs.shape[1] <= 1:
            return torch.tensor(0.0, device=outputs.device)

        centered = outputs - outputs.mean(dim=0)
        cov = (centered.T @ centered) / (outputs.shape[0] - 1)
        mask = ~torch.eye(cov.size(0), device=cov.device).bool()
        return (cov[mask] ** 2).mean()


class SmallEntropyLoss(PretrainingLoss):
    """Binary entropy: push each output toward 0 or 1.

    L = mean( -(p log(p) + (1-p) log(1-p)) )

    Minimizing this encourages decisive/confident predictions,
    useful for boolean/CNF outputs.
    """

    def __init__(self, eps: float = 1e-7):
        self.eps = eps

    def __call__(
        self, outputs: torch.Tensor, inputs: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        p = outputs.clamp(self.eps, 1.0 - self.eps)
        return -(p * p.log() + (1.0 - p) * (1.0 - p).log()).mean()


class LargeEntropyLoss(PretrainingLoss):
    """Push the batch mean of each output dimension toward 0.5.

    L = mean( (p_bar - 0.5)^2 )

    Encourages balanced output distribution (roughly equal 0s and 1s),
    preventing collapse to all-zero or all-one outputs.
    """

    def __call__(
        self, outputs: torch.Tensor, inputs: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        return ((outputs.mean(dim=0) - 0.5) ** 2).mean()


class L1RegularizationLoss(PretrainingLoss):
    """L1 norm over all model parameters."""

    def __call__(
        self, outputs: torch.Tensor, inputs: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        model: nn.Module | None = kwargs.get("model")
        if model is None:
            return torch.tensor(0.0, device=outputs.device)
        return sum(p.abs().sum() for p in model.parameters())


class ConstraintLoss(PretrainingLoss):
    """Push outputs toward constraint satisfaction via fix() projection.

    Detaches outputs, projects them to the nearest satisfying point using
    constraint.fix(), then computes L1 distance. Gradient flows only through
    the original outputs (straight-through style).

    fix() is called once at ``recalc_start`` epoch to populate the cache.
    The loss is zero before that epoch and uses cached targets from then on.

    Requires constraint=<constraint_obj> and epoch=<int> in kwargs.
    """

    def __init__(self, recalc_start: int = 10):
        self.recalc_start = recalc_start
        self._cache: dict[bytes, np.ndarray] = {}

    def __call__(
        self, outputs: torch.Tensor, inputs: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        constraint = kwargs.get("constraint")
        if constraint is None:
            return torch.tensor(0.0, device=outputs.device)

        epoch = kwargs.get("epoch", 0)

        if epoch < self.recalc_start:
            return torch.tensor(0.0, device=outputs.device)

        x_np = inputs.detach().cpu().numpy()
        y_np = outputs.detach().cpu().numpy()

        misses = []
        for i in range(len(x_np)):
            if x_np[i].tobytes() not in self._cache:
                misses.append(i)

        if misses:
            y_fixed = np.stack(constraint.fix(y_np[misses], x_np[misses]))
            for j, i in enumerate(misses):
                self._cache[x_np[i].tobytes()] = y_fixed[j]

        target = torch.tensor(
            np.stack([self._cache[x_np[i].tobytes()] for i in range(len(x_np))]),
            dtype=outputs.dtype,
            device=outputs.device,
        )
        return nn.functional.l1_loss(outputs, target)


LOSS_REGISTRY: dict[str, type[PretrainingLoss]] = {
    "soft_repulsion": SoftRepulsionLoss,
    "decorrelation": DecorrelationLoss,
    "small_entropy": SmallEntropyLoss,
    "large_entropy": LargeEntropyLoss,
    "l1": L1RegularizationLoss,
    "constraint": ConstraintLoss,
}


class CompositeLoss:
    """Weighted sum of multiple PretrainingLoss instances."""

    def __init__(self, components: list[tuple[float, PretrainingLoss]]):
        self.components = components

    def __call__(
        self, outputs: torch.Tensor, inputs: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=outputs.device)
        for weight, loss_fn in self.components:
            total = total + weight * loss_fn(outputs, inputs, **kwargs)
        return total

    def call_detailed(
        self, outputs: torch.Tensor, inputs: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return total loss and per-component breakdown."""
        total = torch.tensor(0.0, device=outputs.device)
        details = {}
        for weight, loss_fn in self.components:
            val = loss_fn(outputs, inputs, **kwargs)
            name = type(loss_fn).__name__
            details[name] = float(val.item())
            total = total + weight * val
        details["total"] = float(total.item())
        return total, details

    def warmup_epochs(self) -> int:
        """Max recalc_start across all losses — early stopping should wait until then."""
        warmup = 0
        for _, loss_fn in self.components:
            if hasattr(loss_fn, "recalc_start"):
                warmup = max(warmup, loss_fn.recalc_start)
        return warmup

    @staticmethod
    def from_configs(loss_configs: list) -> "CompositeLoss":
        """Build CompositeLoss from a list of LossConfig objects."""
        components = []
        for cfg in loss_configs:
            cls = LOSS_REGISTRY[cfg.name]
            loss_fn = cls(**cfg.params)
            components.append((cfg.weight, loss_fn))
        return CompositeLoss(components)
