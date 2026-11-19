"""CNF (Conjunctive Normal Form) constraint over V = I ∪ O."""

import itertools
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from cnfgen import RandomKCNF
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from pysat.solvers import Solver

from nesy_data_gen.constraints.base import Constraint
from nesy_data_gen.constraints.config import CNFConstraintConfig

FORALL_MAX_INPUT_BITS = 10


class CNFConstraintModule(nn.Module):
    """Differentiable CNF constraint probability module.

    For each sample computes the probability that every clause is satisfied,
    assuming each y_i is an independent Bernoulli(p_i) variable.

    For clause ``(l_1 OR l_2 OR ... OR l_k)``::

        p(clause satisfied) = 1 - prod_j  p(l_j = False)

    where:
        p(positive literal  y_i = False) = 1 - preds[:, i]
        p(negative literal ~y_i = False) = preds[:, i]

    Returns a ``(B,)`` tensor — the product of all clause probabilities.
    """

    def __init__(self, clauses: list, n_outputs: int):
        super().__init__()
        self.n_outputs = n_outputs
        # pre-process clauses: list of (var_idx_0indexed, is_positive) per clause
        self._clauses_info = []
        for clause in clauses:
            lits = []
            for lit in clause:
                if abs(lit) > n_outputs:
                    continue
                lits.append((abs(lit) - 1, lit > 0))
            if lits:
                self._clauses_info.append(lits)

    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        prob = torch.ones(preds.shape[0], device=preds.device)
        for lits in self._clauses_info:
            clause_unsat = torch.ones(preds.shape[0], device=preds.device)
            for var_idx, is_positive in lits:
                p_false = (
                    (1.0 - preds[:, var_idx]) if is_positive else preds[:, var_idx]
                )
                clause_unsat = clause_unsat * p_false
            prob = prob * (1.0 - clause_unsat).clamp(min=1e-8)
        return prob


class CNFConstraint(Constraint, CNFConstraintConfig):
    """CNF constraint over Boolean variables V = I ∪ O.

    SAT variable indices:
        1 .. n_outputs             -> output variables (x_1 .. x_n)
        n_outputs+1 .. n_vars      -> input  variables (i_1 .. i_m)
    """

    def __str__(self) -> str:
        if self.use_input_variables:
            mode = "∀I∃O" if self.require_all_inputs_valid else "∃I∃O"
            vars_desc = f"{self.n_vars} vars [{self.n_outputs}O+{self.n_inputs}I]"
        else:
            mode = "V=O"
            vars_desc = f"{self.n_vars} vars [{self.n_outputs}O]"
        return (
            f"CNF({self.k}-SAT, {self.n_clauses} clauses, "
            f"{vars_desc}, {mode}, {self.c}x{self.b} regions)"
        )

    def _var_name(self, var_idx: int) -> str:
        if var_idx <= self.n_outputs:
            return f"x{var_idx}"
        return f"i{var_idx - self.n_outputs}"

    def _to_constraint_module(self, constraint_idx: int) -> "CNFConstraintModule":
        """Return a differentiable CNFConstraintModule for use as a semantic loss."""
        constraint = self._constraints[constraint_idx]
        return CNFConstraintModule(constraint._clauses, self.n_outputs)

    def _to_pishield_str_one(self, constraint_idx: int) -> str:
        """Convert CNF clauses to PiShield propositional 'or' format.

        Output variables (DIMACS indices 1..n_outputs) become y_0..y_{n-1}.
        Input variables are dropped — PiShield only corrects output variables.
        One clause per line, e.g.::

            y_0 or not y_1 or y_2
            not y_0 or not y_3
        """
        constraint = self._constraints[constraint_idx]
        lines = []
        for clause in constraint._clauses:
            parts = []
            for lit in clause:
                if abs(lit) > self.n_outputs:
                    continue
                var_idx = abs(lit) - 1  # 0-indexed
                parts.append(f"y_{var_idx}" if lit > 0 else f"not y_{var_idx}")
            if parts:
                lines.append(" or ".join(parts))
        return "\n".join(lines)

    def _to_str_one(self, constraint: Any) -> str:
        clause_strs = []
        for clause in constraint._clauses:
            literals = []
            for lit in clause:
                name = self._var_name(abs(lit))
                literals.append(name if lit > 0 else f"¬{name}")
            clause_strs.append(f"({' ∨ '.join(literals)})")
        return "\n  ∧ ".join(clause_strs)

    def _check_one(self, vector: Any, input: np.ndarray, constraint: Any) -> bool:
        with Solver(bootstrap_with=constraint._clauses) as solver:
            assumptions = []
            for i, val in enumerate(vector, start=1):
                assumptions.append(i if val >= 0.5 else -i)
            if self.use_input_variables:
                for j, val in enumerate(input, start=self.n_outputs + 1):
                    assumptions.append(j if val >= 0.5 else -j)
            return solver.solve(assumptions=assumptions) is True

    def _generate_single(
        self, constraint_idx: int, max_retries: int = 1000, seed_offset: int = 0
    ) -> Any:
        for attempt in range(max_retries):
            seed = self.seed + constraint_idx * 10000 + attempt * 1000 + seed_offset
            random.seed(seed)
            assignment = [
                i if random.random() > 0.5 else -i for i in range(1, self.n_vars + 1)
            ]
            cnf = RandomKCNF(
                self.k,
                self.n_vars,
                self.n_clauses,
                seed=seed,
                planted_assignments=[assignment],
            )

            if not self.use_input_variables or not self.require_all_inputs_valid:
                with Solver(bootstrap_with=cnf.clauses()) as solver:
                    if solver.solve():
                        return cnf
            else:
                if self._verify_forall(cnf):
                    return cnf

            if attempt > 0 and attempt % 10 == 0:
                print(f"CNF generation: attempt {attempt} failed, retrying...")

        raise RuntimeError(
            f"Failed to generate CNF satisfying the required mode after "
            f"{max_retries} attempts (constraint_idx={constraint_idx})"
        )

    def _verify_forall(self, cnf) -> bool:
        """Check ∀I ∃O : CNF(I, O) by enumerating all Boolean input assignments."""
        if self.n_inputs > FORALL_MAX_INPUT_BITS:
            warnings.warn(
                f"ForAll verification with {self.n_inputs} input bits "
                f"(2^{self.n_inputs} checks) may be very slow. "
                f"Consider using require_all_inputs_valid=False.",
                stacklevel=2,
            )

        input_var_indices = list(range(self.n_outputs + 1, self.n_vars + 1))

        with Solver(bootstrap_with=cnf.clauses()) as solver:
            for bits in itertools.product([0, 1], repeat=self.n_inputs):
                assumptions = [
                    var if bit else -var for var, bit in zip(input_var_indices, bits)
                ]
                if not solver.solve(assumptions=assumptions):
                    return False
        return True

    def _fix_one(
        self, vector: np.ndarray, input: np.ndarray, constraint: Any
    ) -> np.ndarray:
        """Fix outputs via MaxSAT while keeping inputs fixed."""
        formula = WCNF()
        for clause in constraint._clauses:
            formula.append(list[Any](clause), weight=None)

        if self.use_input_variables:
            for j, val in enumerate(input, start=self.n_outputs + 1):
                formula.append([j if val >= 0.5 else -j], weight=None)

        # Soft clauses: prefer matching original output bits
        for i, val in enumerate(vector, start=1):
            formula.append([i if val >= 0.5 else -i], weight=1)

        with RC2(formula) as rc2:
            model = rc2.compute()

        if model is None:
            raise ValueError("No satisfying assignment found")

        fixed_vec = np.zeros(self.n_outputs, dtype=np.float32)
        for lit in model:
            var = abs(lit)
            if var <= self.n_outputs:
                fixed_vec[var - 1] = 1.0 if lit > 0 else 0.0
        return fixed_vec

    def _save_constraints(self, dir_path) -> None:
        cnf_dir = Path(dir_path) / "cnf"
        cnf_dir.mkdir(exist_ok=True)

        for idx, cnf_obj in self._constraints.items():
            with open(cnf_dir / f"{idx}.cnf", "w") as f:
                f.write(cnf_obj.to_dimacs())

    def _load_constraints(self, dir_path) -> None:
        from cnfgen import CNF as CnfgenCNF
        from pysat.formula import CNF as PysatCNF

        cnf_dir = Path(dir_path) / "cnf"
        self._constraints = {}

        for cnf_file in sorted(cnf_dir.glob("*.cnf")):
            idx = int(cnf_file.stem)
            parsed = PysatCNF(from_file=str(cnf_file))
            self._constraints[idx] = CnfgenCNF([list(c) for c in parsed.clauses])
