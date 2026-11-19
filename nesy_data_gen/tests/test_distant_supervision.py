"""Tests that distant supervision labels are consistent with constraint satisfaction."""

import numpy as np
import pytest

from nesy_data_gen.constraints import CNFConstraint
from nesy_data_gen.generators import DatasetGeneratorFiltering


def _split_ds_labels(Y, n_outputs):
    """Split Y into output columns and the distant supervision label column."""
    y_outputs = Y[:, :n_outputs]
    labels = Y[:, n_outputs]
    pos_mask = labels == 1.0
    neg_mask = labels == 0.0
    return y_outputs, labels, pos_mask, neg_mask


class TestDistantSupervisionConsistency:
    """Verify positive samples satisfy the constraint and negative samples violate it."""

    N_OUTPUTS = 4
    N_INPUTS = 4

    def _make_constraint(self, seed=42):
        cnf = CNFConstraint(n_outputs=self.N_OUTPUTS, c=2, b=1, k=3, r=2.0, seed=seed)
        cnf.generate()
        return cnf

    def _make_filtering_generator(self):
        return DatasetGeneratorFiltering(
            n_inputs=self.N_INPUTS,
            n_outputs=self.N_OUTPUTS,
            n_samples=50,
            epochs=500,
            lr=0.001,
            exploration_set_size=1000,
            distant_supervision=True,
        )

    def _assert_labels_consistent(self, result, constraint):
        X, Y = result["X"], result["Y"]

        assert (
            Y.shape[1] == self.N_OUTPUTS + 1
        ), f"Expected {self.N_OUTPUTS + 1} columns (outputs + label), got {Y.shape[1]}"

        y_outputs, labels, pos_mask, neg_mask = _split_ds_labels(Y, self.N_OUTPUTS)

        assert np.all(pos_mask | neg_mask), "Labels must be 0.0 or 1.0"

        assert pos_mask.sum() > 0, "No positive samples found"
        assert neg_mask.sum() > 0, "No negative samples found"

        pos_satisfied = constraint.check(y_outputs[pos_mask], X[pos_mask])
        assert all(pos_satisfied), (
            f"{sum(not s for s in pos_satisfied)}/{pos_mask.sum()} positive samples "
            "do NOT satisfy the constraint"
        )

        neg_satisfied = constraint.check(y_outputs[neg_mask], X[neg_mask])
        assert not any(neg_satisfied), (
            f"{sum(neg_satisfied)}/{neg_mask.sum()} negative samples "
            "DO satisfy the constraint"
        )

    def test_filtering_distant_supervision_consistency(self):
        """Filtering generator: positive=satisfying, negative=violating."""
        cnf = self._make_constraint(seed=42)
        generator = self._make_filtering_generator()
        result = generator.generate_dataset(cnf, seed=42)
        self._assert_labels_consistent(result, cnf)

    @pytest.mark.parametrize("seed", [1, 42, 123, 999])
    def test_filtering_multiple_seeds(self, seed):
        """Consistency holds across different random seeds."""
        cnf = self._make_constraint(seed=seed)
        generator = self._make_filtering_generator()
        result = generator.generate_dataset(cnf, seed=seed)
        self._assert_labels_consistent(result, cnf)

    def test_balanced_positive_negative(self):
        """Distant supervision should produce balanced positive/negative counts."""
        cnf = self._make_constraint(seed=42)
        generator = self._make_filtering_generator()
        result = generator.generate_dataset(cnf, seed=42)

        _, labels, pos_mask, neg_mask = _split_ds_labels(result["Y"], self.N_OUTPUTS)
        assert pos_mask.sum() == neg_mask.sum(), (
            f"Expected balanced dataset, got {pos_mask.sum()} positive "
            f"and {neg_mask.sum()} negative"
        )
