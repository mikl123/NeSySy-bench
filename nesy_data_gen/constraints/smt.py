"""Base SMT constraint: CNF backbone + abstract G function over real V = I ∪ O."""

import random
import re
from abc import abstractmethod
from typing import Any

import numpy as np
from z3 import (
    And,
    Exists,
    ForAll,
    Implies,
    Optimize,
    Or,
    Real,
    RealVal,
    Solver,
    Sum,
    is_and,
    is_or,
    is_true,
    sat,
    simplify,
    substitute,
)

from nesy_data_gen.constants import FLOAT_PRECISION
from nesy_data_gen.constraints.base import Constraint
from nesy_data_gen.constraints.cnf import CNFConstraint
from nesy_data_gen.constraints.config import SMTConstraintConfig


class SMTConstraint(Constraint, SMTConstraintConfig):
    """Abstract base for SMT constraints over real V = I ∪ O.

    Algorithm:
        1. Generate CNF backbone over |V| Boolean variables.
        2. For each literal occurrence, replace with _apply_g (subclass hook).
        3. Verify satisfiability (ForAll or Exists mode).
        4. Retry with different random parameters on failure.

    Z3 variable naming:
        x1 .. x_{n_outputs}  = output variables
        i1 .. i_{n_inputs}   = input variables
    """

    _cnf_constraint: CNFConstraint | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        backbone_n_inputs = self.n_inputs if self.use_input_variables else 0
        object.__setattr__(
            self,
            "_cnf_constraint",
            CNFConstraint(
                seed=self.seed,
                c=self.c,
                b=self.b,
                n_inputs=backbone_n_inputs,
                n_outputs=self.n_outputs,
                use_input_variables=self.use_input_variables,
                r=self.r,
                k=self.k,
                require_all_inputs_valid=False,
            ),
        )

    @abstractmethod
    def _apply_g(
        self,
        lit: int,
        z3_var: Any,
        z3_out_vars: dict[int, Any],
        z3_inp_vars: dict[int, Any],
    ) -> Any:
        """Replace a single CNF literal with a real-valued Z3 expression.

        Args:
            lit: Signed literal from the CNF clause.
            z3_var: Z3 Real variable corresponding to abs(lit).
            z3_out_vars: {1..n_outputs} -> Z3 Real for output variables.
            z3_inp_vars: {1..n_inputs} -> Z3 Real for input variables.

        Returns:
            Z3 expression replacing this literal.
        """
        pass

    def _cnf_var_to_z3(self, cnf_var_idx: int, z3_out_vars, z3_inp_vars):
        """Map a 1-based CNF variable index to its Z3 Real variable."""
        if cnf_var_idx <= self.n_outputs:
            return z3_out_vars[cnf_var_idx]
        return z3_inp_vars[cnf_var_idx - self.n_outputs]

    def _generate_single(self, constraint_idx: int, max_retries: int = 300) -> Any:

        for attempt in range(max_retries):
            cnf = self._cnf_constraint._generate_single(
                constraint_idx, seed_offset=attempt * 7919
            )

            random.seed(self.seed + constraint_idx * 10000 + attempt * 1000)

            z3_out_vars: dict[int, Any] = {}
            z3_inp_vars: dict[int, Any] = {}
            smt_clauses = []

            for var in range(1, self.n_outputs + 1):
                r = Real(f"x{var}")
                z3_out_vars[var] = r
                smt_clauses.append(And(r >= 0, r <= 1))

            if self.use_input_variables:
                for var in range(1, self.n_inputs + 1):
                    r = Real(f"i{var}")
                    z3_inp_vars[var] = r
                    smt_clauses.append(And(r >= 0, r <= 1))

            for clause in cnf._clauses:
                z3_clause = []
                for lit in clause:
                    z3_var = self._cnf_var_to_z3(abs(lit), z3_out_vars, z3_inp_vars)
                    z3_clause.append(
                        self._apply_g(lit, z3_var, z3_out_vars, z3_inp_vars)
                    )
                smt_clauses.append(Or(*z3_clause))

            constraint_formula = And(*smt_clauses)
            solver = Solver()

            if self.use_input_variables:
                input_vars = list(z3_inp_vars.values())
                output_vars = list(z3_out_vars.values())
                input_bounds = [And(v >= 0, v <= 1) for v in input_vars]

                if self.require_all_inputs_valid:
                    solver.add(
                        ForAll(
                            input_vars,
                            Implies(
                                And(*input_bounds),
                                Exists(output_vars, constraint_formula),
                            ),
                        )
                    )
                else:
                    solver.add(
                        Exists(
                            input_vars + output_vars,
                            And(*input_bounds, constraint_formula),
                        )
                    )
            else:
                solver.add(constraint_formula)

            if solver.check() == sat:
                return constraint_formula

            if attempt > 0 and attempt % 10 == 0:
                print(f"SMT generation: attempt {attempt} failed, retrying...")

        raise RuntimeError(
            f"Failed to generate satisfiable SMT constraint after "
            f"{max_retries} attempts (constraint_idx={constraint_idx})"
        )

    def _to_str_one(self, constraint: Any) -> str:
        def format_number(s: str) -> str:
            def replace_fraction(match):
                num, denom = match.groups()
                return f"{float(num) / float(denom):.3f}"

            return re.sub(r"(\d+)/(\d+)", replace_fraction, s)

        def format_expr(s: str) -> str:
            s = str(s).replace(">=", " ≥ ").replace("<=", " ≤ ")
            return format_number(s)

        lines = []
        children = constraint.children() if is_and(constraint) else [constraint]

        for child in children:
            if is_or(child):
                or_parts = []
                for c in child.children():
                    if is_and(c):
                        and_parts = [format_expr(p) for p in c.children()]
                        or_parts.append(f"({' ∧ '.join(and_parts)})")
                    else:
                        or_parts.append(format_expr(c))
                lines.append(f"({' ∨ '.join(or_parts)})")
            elif is_and(child):
                parts = child.children()
                if len(parts) == 2:
                    s1, s2 = format_expr(parts[0]), format_expr(parts[1])
                    lines.append(f"{s1} ∧ {s2}")
                else:
                    lines.append(format_expr(child))
            else:
                lines.append(format_expr(child))

        return "\n  ∧ ".join(lines)

    @staticmethod
    def _relax_inequalities(expr: Any, eps: float) -> Any:
        """Relax all <= and >= in the formula by eps to handle float precision.

        a <= b  becomes  a <= b + eps
        a >= b  becomes  a >= b - eps
        """
        if is_and(expr):
            return And(
                *[SMTConstraint._relax_inequalities(c, eps) for c in expr.children()]
            )
        if is_or(expr):
            return Or(
                *[SMTConstraint._relax_inequalities(c, eps) for c in expr.children()]
            )
        try:
            name = expr.decl().name()
        except Exception:
            return expr
        if name == "<=":
            return expr.arg(0) <= expr.arg(1) + eps
        if name == ">=":
            return expr.arg(0) >= expr.arg(1) - eps
        return expr

    def _check_one(self, vector: Any, input: np.ndarray, constraint: Any) -> bool:
        eps = FLOAT_PRECISION
        relaxed = self._relax_inequalities(constraint, eps)

        subs = []
        for i, val in enumerate(vector, start=1):
            subs.append((Real(f"x{i}"), RealVal(str(float(val)))))
        if self.use_input_variables:
            for i, val in enumerate(input, start=1):
                subs.append((Real(f"i{i}"), RealVal(str(float(val)))))

        grounded = substitute(relaxed, *subs)
        return is_true(simplify(grounded))

    def _fix_one(self, vector: Any, input: np.ndarray, constraint: Any) -> np.ndarray:
        # Ground input variables into the formula up front so Z3 optimizes over
        # outputs only, avoiding redundant equality constraints.
        if self.use_input_variables:
            input_subs = [
                (Real(f"i{i}"), RealVal(str(float(val))))
                for i, val in enumerate(input, start=1)
            ]
            constraint = simplify(substitute(constraint, *input_subs))

        opt = Optimize()
        opt.add(constraint)

        t_vars = []
        var_names = [f"x{i}" for i in range(1, len(vector) + 1)]
        for var_name, target_value in zip(var_names, vector):
            z3_var = Real(var_name)
            t = Real(f"_t_{var_name}")
            target = RealVal(str(target_value))
            opt.add(t >= 0)
            opt.add(t >= z3_var - target)
            opt.add(t >= target - z3_var)
            t_vars.append(t)
        opt.minimize(Sum(t_vars))

        if opt.check() == sat:
            m = opt.model()
            return [float(m.eval(Real(v)).as_fraction()) for v in var_names]
        raise ValueError("Cannot find satisfiable assignment")

    def _save_constraints(self, dir_path) -> None:
        from pathlib import Path

        smt_dir = Path(dir_path) / "smt2"
        smt_dir.mkdir(exist_ok=True)

        for idx, formula in self._constraints.items():
            s = Solver()
            s.add(formula)
            with open(smt_dir / f"{idx}.smt2", "w") as f:
                f.write(s.to_smt2())

    def _load_constraints(self, dir_path) -> None:
        from pathlib import Path

        from z3 import parse_smt2_string

        smt_dir = Path(dir_path) / "smt2"
        self._constraints = {}

        for smt_file in sorted(smt_dir.glob("*.smt2")):
            idx = int(smt_file.stem)
            with open(smt_file) as f:
                smt2_str = f.read()
            cleaned = "\n".join(
                line for line in smt2_str.splitlines() if line.strip() != "(check-sat)"
            )
            assertions = parse_smt2_string(cleaned)
            self._constraints[idx] = (
                And(*assertions) if len(assertions) > 1 else assertions[0]
            )
