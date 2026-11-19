"""Tests for constraint classes."""

import itertools

import numpy as np
import pytest
from pysat.solvers import Solver as SATSolver
from z3 import And, Or, Real, RealVal, Solver, sat, substitute

from nesy_data_gen.constraints import (
    CNFConstraint,
    SMTBoundedIntervalConstraint,
    SMTLinearInequalityConstraint,
)
from nesy_data_gen.constraints.smt import SMTConstraint


@pytest.fixture
def seed():
    """Default random seed for reproducibility."""
    return 42


@pytest.fixture
def small_cnf_params():
    """Small parameter set for CNF tests (exists mode, with inputs)."""
    return {
        "n_inputs": 3,
        "n_outputs": 4,
        "use_input_variables": True,
        "c": 2,
        "b": 1,
        "seed": 42,
        "r": 2.0,
        "require_all_inputs_valid": False,
    }


@pytest.fixture
def small_cnf_forall_params():
    """Small parameter set for CNF tests (forall mode, with inputs)."""
    return {
        "n_inputs": 3,
        "n_outputs": 4,
        "use_input_variables": True,
        "c": 2,
        "b": 1,
        "seed": 42,
        "r": 1.5,
        "require_all_inputs_valid": True,
    }


@pytest.fixture
def small_smt_region_params():
    """Small parameter set for SMT bounded-interval tests (with inputs)."""
    return {
        "n_inputs": 4,
        "n_outputs": 4,
        "use_input_variables": True,
        "c": 2,
        "b": 1,
        "seed": 42,
        "r": 0.5,
        "require_all_inputs_valid": True,
    }


@pytest.fixture
def small_smt_complex_params():
    """Small parameter set for SMT linear inequality tests (with inputs)."""
    return {
        "n_inputs": 4,
        "n_outputs": 4,
        "use_input_variables": True,
        "c": 2,
        "b": 1,
        "seed": 42,
        "r": 1.0,
        "coef_abs_max": 3,
        "n_terms_min": 1,
        "n_terms_max": 4,
        "require_all_inputs_valid": True,
    }


@pytest.fixture
def sample_inputs():
    """Sample real input array for testing."""
    np.random.seed(42)
    return np.random.rand(100, 4)


@pytest.fixture
def sample_outputs():
    """Sample real output array for testing."""
    np.random.seed(42)
    return np.random.rand(100, 4)


@pytest.fixture
def sample_binary_inputs():
    """Sample binary input array for testing (0 or 1)."""
    np.random.seed(41)
    return np.random.randint(0, 2, size=(100, 3)).astype(np.float32)


@pytest.fixture
def sample_binary_outputs():
    """Sample binary output array for testing (0 or 1)."""
    np.random.seed(42)
    return np.random.randint(0, 2, size=(100, 4))


class TestCNFConstraint:
    """Tests for CNFConstraint over V = I ∪ O."""

    def test_create_constraint(self, small_cnf_params):
        cnf = CNFConstraint(**small_cnf_params)
        assert cnf.n_outputs == 4
        assert cnf.n_inputs == 3
        assert cnf.n_vars == 7
        assert cnf.k == 3

    def test_generate_exists_mode(self, small_cnf_params):
        cnf = CNFConstraint(**small_cnf_params)
        cnf.generate()
        assert len(cnf._constraints) == small_cnf_params["c"]

    def test_generate_forall_mode(self, small_cnf_forall_params):
        cnf = CNFConstraint(**small_cnf_forall_params)
        cnf.generate()
        assert len(cnf._constraints) == small_cnf_forall_params["c"]

    def test_fix_keeps_inputs_and_satisfies(
        self, small_cnf_forall_params, sample_binary_inputs, sample_binary_outputs
    ):
        """Fixed outputs satisfy the constraint while inputs stay fixed."""
        cnf = CNFConstraint(**small_cnf_forall_params)
        cnf.generate()
        fixed = cnf.fix(sample_binary_outputs, sample_binary_inputs)
        assert len(fixed) == len(sample_binary_outputs)
        result = cnf.check(fixed, sample_binary_inputs)
        assert all(result)

    def test_forall_satisfiable_for_all_inputs(self, small_cnf_forall_params):
        """In forall mode every input assignment must have a satisfying output."""
        cnf = CNFConstraint(**small_cnf_forall_params)
        cnf.generate()

        for constraint_data in cnf._constraints.values():
            input_var_indices = list(range(cnf.n_outputs + 1, cnf.n_vars + 1))
            for bits in itertools.product([0, 1], repeat=cnf.n_inputs):
                assumptions = [v if b else -v for v, b in zip(input_var_indices, bits)]
                with SATSolver(bootstrap_with=constraint_data._clauses) as s:
                    assert s.solve(
                        assumptions=assumptions
                    ), f"ForAll violated for input bits {bits}"

    def test_output_only_n_vars(self):
        """With use_input_variables=False, n_vars should equal n_outputs."""
        cnf = CNFConstraint(
            n_outputs=6, n_inputs=4, use_input_variables=False, seed=42, r=2.0
        )
        assert cnf.n_vars == 6

    def test_output_only_generate_and_check(self):
        """Output-only CNF should generate and check without input variables."""
        cnf = CNFConstraint(
            n_outputs=6,
            n_inputs=0,
            use_input_variables=False,
            c=2,
            b=1,
            seed=42,
            r=2.0,
        )
        cnf.generate()
        assert len(cnf._constraints) == 2
        outputs = np.random.randint(0, 2, size=(50, 6)).astype(np.float32)
        inputs = np.random.rand(50, 1)
        fixed = cnf.fix(outputs, inputs)
        result = cnf.check(fixed, inputs)
        assert all(result)


class TestSMTBoundedIntervalConstraint:
    """Tests for SMTBoundedIntervalConstraint (bounded-interval G)."""

    def test_create_constraint(self, small_smt_region_params):
        smt = SMTBoundedIntervalConstraint(**small_smt_region_params)
        assert smt.n_outputs == 4
        assert smt.n_inputs == 4
        assert smt.n_vars == 8

    def test_generate_constraint(self, small_smt_region_params):
        smt = SMTBoundedIntervalConstraint(**small_smt_region_params)
        smt.generate()
        assert len(smt._constraints) == small_smt_region_params["c"]

        for constraint in smt._constraints.values():
            solver = Solver()
            solver.add(constraint)
            assert solver.check() == sat

    def test_fix_satisfies_constraint(
        self, small_smt_region_params, sample_inputs, sample_outputs
    ):
        smt = SMTBoundedIntervalConstraint(**small_smt_region_params)
        smt.generate()
        fixed = smt.fix(sample_outputs, sample_inputs)
        assert len(fixed) == len(sample_outputs)
        result = smt.check(fixed, sample_inputs)
        assert all(result)

    def test_output_only_generate(self):
        """Output-only bounded-interval should generate satisfiable constraints."""
        smt = SMTBoundedIntervalConstraint(
            n_outputs=4,
            n_inputs=0,
            use_input_variables=False,
            c=1,
            b=1,
            seed=42,
            r=0.5,
        )
        assert smt.n_vars == 4
        smt.generate()
        assert len(smt._constraints) == 1
        for constraint in smt._constraints.values():
            solver = Solver()
            solver.add(constraint)
            assert solver.check() == sat
        outputs = np.random.rand(50, 4).astype(np.float32)
        inputs = np.random.rand(50, 1)
        fixed = smt.fix(outputs, inputs)
        result = smt.check(fixed, inputs)
        assert all(result)


class TestSMTLinearInequalityConstraint:
    """Tests for SMTLinearInequalityConstraint (linear inequality G)."""

    def test_create_constraint(self, small_smt_complex_params):
        smt = SMTLinearInequalityConstraint(**small_smt_complex_params)
        assert smt.n_outputs == 4
        assert smt.n_inputs == 4
        assert smt.n_vars == 8

    def test_generate_constraint(self, small_smt_complex_params):
        smt = SMTLinearInequalityConstraint(**small_smt_complex_params)
        smt.generate()
        assert len(smt._constraints) == small_smt_complex_params["c"]

        for constraint in smt._constraints.values():
            solver = Solver()
            solver.add(constraint)
            assert solver.check() == sat

    def test_fix_satisfies_constraint(
        self, small_smt_complex_params, sample_inputs, sample_outputs
    ):
        smt = SMTLinearInequalityConstraint(**small_smt_complex_params)
        smt.generate()
        fixed = smt.fix(sample_outputs, sample_inputs)
        assert len(fixed) == len(sample_outputs)
        result = smt.check(fixed, sample_inputs)
        assert all(result)

    def test_output_only_generate(self):
        """Output-only linear inequality should generate satisfiable constraints."""
        smt = SMTLinearInequalityConstraint(
            n_outputs=4,
            n_inputs=0,
            use_input_variables=False,
            c=1,
            b=1,
            seed=42,
            r=1.0,
            coef_abs_max=3,
            n_terms_min=1,
            n_terms_max=3,
        )
        assert smt.n_vars == 4
        smt.generate()
        assert len(smt._constraints) == 1
        for constraint in smt._constraints.values():
            solver = Solver()
            solver.add(constraint)
            assert solver.check() == sat
        outputs = np.random.rand(50, 4).astype(np.float32)
        inputs = np.random.rand(50, 1)
        fixed = smt.fix(outputs, inputs)
        result = smt.check(fixed, inputs)
        assert all(result)


class TestRelaxInequalities:
    """Unit tests for SMTConstraint._relax_inequalities."""

    def test_le_relaxed(self):
        """a <= 0.5 with a=0.50001 fails strict but passes relaxed."""
        x = Real("x")
        eps = 0.001
        expr = x <= 0.5
        relaxed = SMTConstraint._relax_inequalities(expr, eps)
        print(f"  original: {expr}")
        print(f"  relaxed:  {relaxed}")

        strict = substitute(expr, (x, RealVal("0.50001")))
        solver = Solver()
        solver.add(strict)
        assert solver.check() != sat

        ground = substitute(relaxed, (x, RealVal("0.50001")))
        solver = Solver()
        solver.add(ground)
        assert solver.check() == sat

    def test_ge_relaxed(self):
        """a >= 0.5 with a=0.49999 fails strict but passes relaxed."""
        x = Real("x")
        eps = 0.001
        expr = x >= 0.5
        relaxed = SMTConstraint._relax_inequalities(expr, eps)
        print(f"  original: {expr}")
        print(f"  relaxed:  {relaxed}")

        strict = substitute(expr, (x, RealVal("0.49999")))
        solver = Solver()
        solver.add(strict)
        assert solver.check() != sat

        ground = substitute(relaxed, (x, RealVal("0.49999")))
        solver = Solver()
        solver.add(ground)
        assert solver.check() == sat

    def test_and_or_recursion(self):
        """Relaxation recurses through And/Or."""
        x, y = Real("x"), Real("y")
        eps = 0.01
        expr = And(Or(x <= 0.3, y >= 0.7), x >= 0.1)
        relaxed = SMTConstraint._relax_inequalities(expr, eps)
        print(f"  original: {expr}")
        print(f"  relaxed:  {relaxed}")

        ground = substitute(relaxed, (x, RealVal("0.099")), (y, RealVal("0.691")))
        solver = Solver()
        solver.add(ground)
        assert solver.check() == sat

    def test_exact_boundary_passes(self):
        """Value exactly on the boundary passes both strict and relaxed."""
        x = Real("x")
        eps = 0.001
        expr = x <= 0.5
        relaxed = SMTConstraint._relax_inequalities(expr, eps)
        print(f"  original: {expr}")
        print(f"  relaxed:  {relaxed}")

        ground = substitute(relaxed, (x, RealVal("0.5")))
        solver = Solver()
        solver.add(ground)
        assert solver.check() == sat
