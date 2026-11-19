"""Pydantic configuration models for dataset generators."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LossConfig(BaseModel):
    """Configuration for a single pretraining loss component.

    Available losses (see LOSS_REGISTRY in losses.py):
        - "soft_repulsion": Soft 1/(dist+eps) repulsion (default). params: {eps: float}
        - "decorrelation": Penalizes correlated output dimensions. params: {}
        - "small_entropy": Binary entropy, pushes outputs toward 0 or 1. params: {eps: float}
        - "large_entropy": Pushes batch mean of each dimension toward 0.5. params: {}
        - "uniform_spread": Penalizes non-uniform spacing per dimension. params: {}
        - "l1": L1 regularization over model parameters. params: {}
        - "constraint": ||y - fix(stopgrad(y), x)||, pushes outputs toward constraint satisfaction. params: {}
    """

    name: str = Field(description="Registered loss name (e.g. 'diversity', 'l1')")
    weight: float = Field(
        default=1.0, description="Multiplicative weight for this loss"
    )
    params: dict = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the loss constructor",
    )


class BaseGeneratorConfig(BaseModel):
    """Shared configuration for all dataset generators."""

    n_inputs: int = Field(default=4, ge=1, description="Number of input features")
    n_outputs: int = Field(default=4, ge=1, description="Number of output features")
    n_samples: int = Field(
        default=500, ge=1, description="Total number of samples to generate"
    )
    hidden_dim: int = Field(
        default=64, ge=1, description="Number of neurons in a hidden layer"
    )
    num_hidden: int = Field(default=2, ge=1, description="Number of hidden layers")
    weight_decay: float = Field(default=0.0, ge=0, description="Weight decay")
    epochs: int = Field(
        default=1000, ge=1, description="Training epochs for diversity model"
    )
    lr: float = Field(default=0.001, gt=0, description="Learning rate")
    batch_size: int = Field(
        default=512, ge=1, description="Batch size for diversity model training"
    )
    exploration_set_size: int = Field(
        default=10000, ge=1, description="Size of initial exploration set"
    )
    losses: list[LossConfig] = Field(
        default_factory=lambda: [LossConfig(name="soft_repulsion", weight=1.0)],
        description="Pretraining loss components with weights. "
        "Default: soft repulsion loss only.",
    )
    exploration_points_csv_path: Optional[str] = Field(
        default=None,
        description="Optional CSV path to save all exploration inputs and pretrained outputs",
    )
    distant_supervision: bool = Field(
        default=False,
        description="When True, also collect non-satisfying samples and append a "
        "constraint-satisfaction label column to Y. Only for non-real constraints.",
    )


class DatasetGeneratorConfig(BaseGeneratorConfig):
    """Configuration for DatasetGeneratorFiltering."""

    model_config = ConfigDict(extra="forbid", strict=True)
    max_iterations: int = Field(
        default=3,
        ge=1,
        description="Maximum number of iterations to sample near boundary points",
    )
    noise_std: float = Field(
        default=0.2,
        gt=0.0,
        description="Noise std when sampling near satisfying points",
    )


class FixGuidedGeneratorConfig(BaseGeneratorConfig):
    """Configuration for DatasetGeneratorFixGuided."""

    model_config = ConfigDict(extra="forbid", strict=True)
    fix_noise_std: float = Field(
        default=0.3,
        gt=0.0,
        description="Initial noise std when sampling near fixed boundary points",
    )
    fix_iterations: int = Field(
        default=20,
        ge=1,
        description="Max rounds of sampling interior points near boundary seeds",
    )
