"""Custom constraint — subclass `Constraint` with your own predicate."""

import sys
from pathlib import Path

import numpy as np
from pydantic import ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesy_data_gen.constraints import Constraint
from nesy_data_gen.constraints.config import BaseConstraintConfig


class ThresholdSumConstraintConfig(BaseConstraintConfig):
    model_config = ConfigDict(extra="forbid", strict=True)
    threshold_min: float = Field(
        default=0.0, description="Lower end of sampled threshold range"
    )
    threshold_max: float = Field(
        default=1.0, description="Upper end of sampled threshold range"
    )


class ThresholdSumConstraint(Constraint, ThresholdSumConstraintConfig):
    """Sum of output variables must be >= a per-constraint random threshold."""

    def _generate_single(self, constraint_idx: int) -> float:
        rng = np.random.default_rng(self.seed + constraint_idx)
        return float(rng.uniform(self.threshold_min, self.threshold_max))

    def _check_one(self, vector, input, constraint) -> bool:
        return float(np.sum(vector)) >= constraint

    def _to_str_one(self, constraint) -> str:
        return f"sum(x) >= {constraint:.3f}"

    def _fix_one(self, vector, input, constraint):
        v = np.asarray(vector, dtype=np.float32).copy()
        deficit = constraint - float(v.sum())
        if deficit <= 0:
            return v
        return np.clip(v + deficit / len(v), 0.0, 1.0)


if __name__ == "__main__":
    # c=2 unique constraints over b=1 region each -> two regions with different thresholds.
    constraint = ThresholdSumConstraint(
        n_outputs=3, threshold_min=0.5, threshold_max=2.5, c=2, b=1, seed=0
    )
    constraint.generate()

    print("formulas:", constraint.to_str())

    outputs = np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]], dtype=np.float32)
    # First feature picks the region (R0 = [0.0, 0.5], R1 = [0.5, 1.0]).
    inputs = np.array([[0.2], [0.7]], dtype=np.float32)

    print("check:        ", constraint.check(outputs, inputs))
    fixed = constraint.fix(outputs, inputs)
    print("check (fixed):", constraint.check(fixed, inputs))

    # Generate a dataset against the custom constraint via the standard pipeline.
    from nesy_data_gen.generators import DatasetGeneratorFixGuided

    gen = DatasetGeneratorFixGuided(
        n_inputs=2, n_outputs=3, n_samples=200, exploration_set_size=2000, epochs=30
    )
    dataset = gen.generate_dataset(constraint, seed=0)
    X, Y = dataset["X"], dataset["Y"]
    print(
        f"dataset: X={X.shape}, Y={Y.shape}, all satisfy: {all(constraint.check(Y, X))}"
    )
