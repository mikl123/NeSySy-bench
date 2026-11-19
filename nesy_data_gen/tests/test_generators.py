"""Tests for dataset generators."""

import numpy as np

from nesy_data_gen.constraints import (
    CNFConstraint,
    SMTBoundedIntervalConstraint,
    SMTLinearInequalityConstraint,
)
from nesy_data_gen.generators import (
    DatasetGeneratorFiltering,
    DatasetGeneratorFixGuided,
)


class TestDatasetGeneratorFiltering:
    """Tests for DatasetGeneratorFiltering."""

    def test_create_generator(self):
        """Test generator creation."""
        generator = DatasetGeneratorFiltering(
            n_inputs=4,
            n_outputs=4,
            n_samples=100,
            epochs=10,
            lr=0.01,
            exploration_set_size=500,
        )
        assert generator.n_inputs == 4
        assert generator.n_outputs == 4
        assert generator.n_samples == 100

    def test_generate_cnf_dataset(self):
        """Test dataset generation with CNF constraint."""
        # Create constraint
        cnf = CNFConstraint(n_outputs=4, c=2, b=1, k=3, r=2.0, seed=42)
        cnf.generate()

        # Create generator
        generator = DatasetGeneratorFiltering(
            n_inputs=4,
            n_outputs=4,
            n_samples=50,
            epochs=500,
            lr=0.001,
            exploration_set_size=1000,
        )

        # Generate dataset
        result = generator.generate_dataset(cnf, seed=42)

        assert "X" in result
        assert "Y" in result
        assert "constraint" in result
        assert result["X"].shape[0] == 50
        assert result["Y"].shape[0] == 50
        assert result["X"].shape[1] == 4
        assert result["Y"].shape[1] == 4

    def test_all_cnf_outputs_satisfy_constraint(self):
        """Test that all generated CNF outputs satisfy the constraint."""
        # Create constraint
        cnf = CNFConstraint(n_outputs=4, c=2, b=1, k=3, r=2.0, seed=42)
        cnf.generate()

        # Create generator
        generator = DatasetGeneratorFiltering(
            n_inputs=4,
            n_outputs=4,
            n_samples=50,
            epochs=5000,
            lr=0.001,
            exploration_set_size=1000,
        )

        # Generate dataset
        result = generator.generate_dataset(cnf, seed=42)

        # Check all outputs satisfy constraint
        satisfied = cnf.check(result["Y"], result["X"])
        assert all(satisfied), "Not all generated CNF outputs satisfy the constraint"

    def test_generate_smt_region_dataset(self):
        """Test dataset generation and constraint satisfaction with SMTBoundedInterval."""
        smt_region = SMTBoundedIntervalConstraint(
            n_outputs=4,
            n_inputs=4,
            r=1,
            c=1,
            b=1,
            k=3,
            min_gap=0.3,
            seed=42,
        )
        smt_region.generate()

        generator = DatasetGeneratorFiltering(
            n_inputs=4,
            n_outputs=4,
            n_samples=50,
            epochs=500,
            lr=0.001,
            exploration_set_size=10000,
            losses=[
                {"name": "soft_repulsion", "weight": 1.0},
            ],
        )

        result = generator.generate_dataset(smt_region, seed=42)

        assert result["X"].shape[0] > 0
        assert result["Y"].shape[0] > 0
        assert result["X"].shape[1] == 4
        assert result["Y"].shape[1] == 4

        satisfied = smt_region.check(result["Y"], result["X"])
        assert all(
            satisfied
        ), "Not all generated SMTBoundedInterval outputs satisfy the constraint"

    def test_generate_smt_linear_inequality_dataset(self):
        """Test dataset generation and constraint satisfaction with SMTLinearInequality."""
        smt_complex = SMTLinearInequalityConstraint(
            n_outputs=4,
            n_inputs=4,
            r=2,
            c=2,
            b=1,
            seed=42,
            coef_abs_max=5,
        )
        smt_complex.generate()

        generator = DatasetGeneratorFiltering(
            n_inputs=4,
            n_outputs=4,
            n_samples=50,
            epochs=500,
            lr=0.001,
            exploration_set_size=10000,
        )

        result = generator.generate_dataset(smt_complex, seed=42)

        assert result["X"].shape[0] > 0
        assert result["Y"].shape[0] > 0
        assert result["X"].shape[1] == 4
        assert result["Y"].shape[1] == 4

        satisfied = smt_complex.check(result["Y"], result["X"])
        assert all(
            satisfied
        ), "Not all generated SMTLinearInequality outputs satisfy the constraint"


class TestDatasetGeneratorFixGuided:
    """Tests for DatasetGeneratorFixGuided."""

    def test_create_generator(self):
        """Test generator creation."""
        generator = DatasetGeneratorFixGuided(
            n_inputs=4,
            n_outputs=4,
            n_samples=100,
            epochs=10,
            lr=0.01,
            exploration_set_size=200,
        )
        assert generator.n_inputs == 4
        assert generator.n_outputs == 4
        assert generator.n_samples == 100
        assert generator.exploration_set_size == 200

    def test_generate_cnf_dataset(self):
        """Test dataset generation and constraint satisfaction with CNF."""
        cnf = CNFConstraint(n_outputs=4, c=2, b=1, k=3, r=2.0, seed=42)
        cnf.generate()

        generator = DatasetGeneratorFixGuided(
            n_inputs=4,
            n_outputs=4,
            n_samples=50,
            epochs=100,
            lr=0.01,
        )

        result = generator.generate_dataset(cnf, seed=42)

        assert "X" in result
        assert "Y" in result
        assert "constraint" in result
        assert result["X"].shape[1] == 4
        assert result["Y"].shape[1] == 4

        assert np.isin(result["Y"], [0, 1]).all(), "CNF outputs should be binary"

        satisfied = cnf.check(result["Y"], result["X"])
        assert all(satisfied), "Not all generated CNF outputs satisfy the constraint"

    def test_generate_smt_region_dataset(self):
        """Test dataset generation and constraint satisfaction with SMTBoundedInterval."""
        smt_region = SMTBoundedIntervalConstraint(
            n_outputs=4,
            n_inputs=4,
            c=2,
            b=1,
            k=2,
            r=2.0,
            min_gap=0.1,
            seed=42,
        )
        smt_region.generate()

        generator = DatasetGeneratorFixGuided(
            n_inputs=4,
            n_outputs=4,
            n_samples=50,
            epochs=100,
            lr=0.01,
        )

        result = generator.generate_dataset(smt_region, seed=42)

        assert result["X"].shape[1] == 4
        assert result["Y"].shape[1] == 4

        satisfied = smt_region.check(result["Y"], result["X"])
        assert all(
            satisfied
        ), "Not all generated SMTBoundedInterval outputs satisfy the constraint"

    def test_generate_smt_linear_inequality_dataset(self):
        """Test dataset generation and constraint satisfaction with SMTLinearInequality."""
        smt_complex = SMTLinearInequalityConstraint(
            n_outputs=4,
            n_inputs=4,
            r=2,
            c=2,
            b=1,
            seed=42,
            coef_abs_max=5,
        )
        smt_complex.generate()

        generator = DatasetGeneratorFixGuided(
            n_inputs=4,
            n_outputs=4,
            n_samples=50,
            epochs=100,
            lr=0.01,
        )

        result = generator.generate_dataset(smt_complex, seed=42)

        assert result["X"].shape[1] == 4
        assert result["Y"].shape[1] == 4

        satisfied = smt_complex.check(result["Y"], result["X"])
        assert all(
            satisfied
        ), "Not all generated SMTLinearInequality outputs satisfy the constraint"

    def test_same_seed_reproducible_tiny(self):
        """Same seed for constraint+generator should reproduce the same tiny dataset."""
        seed = 7
        constraint_kwargs = {
            "n_outputs": 4,
            "n_inputs": 4,
            "c": 1,
            "b": 1,
            "k": 3,
            "r": 2.0,
            "seed": seed,
        }
        generator_kwargs = {
            "n_inputs": 4,
            "n_outputs": 4,
            "n_samples": 12,
            "epochs": 5,
            "lr": 0.01,
            "exploration_set_size": 200,
        }

        c1 = CNFConstraint(**constraint_kwargs)
        c1.generate()
        g1 = DatasetGeneratorFixGuided(**generator_kwargs)
        out1 = g1.generate_dataset(c1, seed=seed)

        c2 = CNFConstraint(**constraint_kwargs)
        c2.generate()
        g2 = DatasetGeneratorFixGuided(**generator_kwargs)
        out2 = g2.generate_dataset(c2, seed=seed)

        np.testing.assert_array_equal(out1["X"], out2["X"])
        np.testing.assert_array_equal(out1["Y"], out2["Y"])
