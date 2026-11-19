"""Tests for constraint save_state / load_state roundtrip."""

import numpy as np
import pytest
from z3 import Solver, sat

from nesy_data_gen.constraints import (
    CNFConstraint,
    Constraint,
    SMTBoundedIntervalConstraint,
    SMTLinearInequalityConstraint,
)


@pytest.fixture
def cnf_constraint():
    c = CNFConstraint(n_outputs=6, c=2, b=1, seed=42, r=2.0)
    c.generate()
    return c


@pytest.fixture
def smt_bounded_constraint():
    c = SMTBoundedIntervalConstraint(n_outputs=3, c=2, b=1, seed=42, r=1.0, min_gap=0.1)
    c.generate()
    return c


@pytest.fixture
def smt_linear_inequality_constraint():
    c = SMTLinearInequalityConstraint(
        n_outputs=3,
        c=2,
        b=1,
        seed=42,
        r=1.0,
        coef_abs_max=3,
        n_terms_min=1,
        n_terms_max=3,
    )
    c.generate()
    return c


class TestCNFPersistence:
    def test_save_load_roundtrip(self, cnf_constraint, tmp_path):
        cnf_constraint.save_state(str(tmp_path))
        loaded = Constraint.load_state(str(tmp_path))

        assert type(loaded).__name__ == "CNFConstraint"
        assert loaded.model_dump() == cnf_constraint.model_dump()
        assert np.array_equal(loaded._bounds, cnf_constraint._bounds)
        assert loaded._region_to_constraint == cnf_constraint._region_to_constraint
        assert loaded._region_bounds == cnf_constraint._region_bounds

        for idx in cnf_constraint._constraints:
            orig_clauses = sorted(
                [sorted(c) for c in cnf_constraint._constraints[idx]._clauses]
            )
            loaded_clauses = sorted(
                [sorted(c) for c in loaded._constraints[idx]._clauses]
            )
            assert orig_clauses == loaded_clauses

        rng = np.random.default_rng(99)
        outputs = rng.integers(0, 2, size=(20, 6)).astype(float)
        inputs = rng.random((20, 1))
        assert cnf_constraint.check(outputs, inputs) == loaded.check(outputs, inputs)


class TestSMTBoundedPersistence:
    def test_save_load_roundtrip(self, smt_bounded_constraint, tmp_path):
        smt_bounded_constraint.save_state(str(tmp_path))
        loaded = Constraint.load_state(str(tmp_path))

        assert type(loaded).__name__ == "SMTBoundedIntervalConstraint"
        assert loaded.model_dump() == smt_bounded_constraint.model_dump()
        assert np.array_equal(loaded._bounds, smt_bounded_constraint._bounds)
        assert (
            loaded._region_to_constraint == smt_bounded_constraint._region_to_constraint
        )
        assert loaded._region_bounds == smt_bounded_constraint._region_bounds

        for formula in loaded._constraints.values():
            s = Solver()
            s.add(formula)
            assert s.check() == sat

        rng = np.random.default_rng(99)
        outputs = rng.random((20, 3))
        inputs = rng.random((20, 1))
        assert smt_bounded_constraint.check(outputs, inputs) == loaded.check(
            outputs, inputs
        )


class TestSMTLinearInequalityPersistence:
    def test_save_load_roundtrip(self, smt_linear_inequality_constraint, tmp_path):
        smt_linear_inequality_constraint.save_state(str(tmp_path))
        loaded = Constraint.load_state(str(tmp_path))

        assert type(loaded).__name__ == "SMTLinearInequalityConstraint"
        assert loaded.model_dump() == smt_linear_inequality_constraint.model_dump()
        assert np.array_equal(loaded._bounds, smt_linear_inequality_constraint._bounds)
        assert (
            loaded._region_to_constraint
            == smt_linear_inequality_constraint._region_to_constraint
        )

        for formula in loaded._constraints.values():
            s = Solver()
            s.add(formula)
            assert s.check() == sat

        rng = np.random.default_rng(99)
        outputs = rng.random((20, 3))
        inputs = rng.random((20, 1))
        assert smt_linear_inequality_constraint.check(outputs, inputs) == loaded.check(
            outputs, inputs
        )
