"""SMT Bounded-Interval constraint: bounded-interval G function."""

import random
from typing import Any

from z3 import And

from nesy_data_gen.constraints.config import SMTBoundedIntervalConstraintConfig
from nesy_data_gen.constraints.smt import SMTConstraint


class SMTBoundedIntervalConstraint(SMTConstraint, SMTBoundedIntervalConstraintConfig):
    """Bounded-interval G function: each literal becomes c_min <= x <= c_max.

    Bounds are random constants in [0, 1] separated by at least min_gap.
    Input variables are already part of V via the CNF backbone.
    """

    def __str__(self) -> str:
        if self.use_input_variables:
            mode = "∀I∃O" if self.require_all_inputs_valid else "∃I∃O"
            vars_desc = f"{self.n_outputs}O+{self.n_inputs}I vars"
        else:
            mode = "V=O"
            vars_desc = f"{self.n_outputs}O vars"
        return f"SMTBoundedInterval({vars_desc}, " f"{mode}, {self.c}x{self.b} regions)"

    def _apply_g(
        self,
        lit: int,
        z3_var: Any,
        z3_out_vars: dict[int, Any],
        z3_inp_vars: dict[int, Any],
    ) -> Any:
        left = random.random() * (1 - self.min_gap)
        right = left + self.min_gap + random.random() * (1 - left - self.min_gap)
        return And(left <= z3_var, z3_var <= right)

    def _to_pishield_str_one(self, constraint_idx: int) -> str:
        """Convert a Z3 SMT constraint to DRL text format.

        Walks the stored Z3 expression tree. For OR clauses whose disjuncts
        are AND conjunctions (bounded intervals like ``And(x >= lo, x <= hi)``),
        applies the distributive law to convert to CNF so each DRL line is a
        pure disjunction of linear inequalities.

        Example::

            Or(And(a>=0.1, a<=0.3), And(b>=0.5, b<=0.6))

        becomes four DRL lines (2^2 combinations)::

            y_0>=0.1 or y_1>=0.5
            y_0>=0.1 or -y_1>=-0.6
            -y_0>=-0.3 or y_1>=0.5
            -y_0>=-0.3 or -y_1>=-0.6
        """
        from itertools import product as cartesian_product

        from z3 import is_and, is_le, is_or

        z3_constr = self._constraints[constraint_idx]

        def _z3_num_to_float(num):
            """Convert a Z3 numeric node to Python float.

            Handles both RatNumRef (fresh) and ArithRef (after SMT-LIB2 round-trip).
            """
            if hasattr(num, "numerator_as_long"):
                return float(num.numerator_as_long()) / float(num.denominator_as_long())
            from z3 import simplify

            s = simplify(num)
            if hasattr(s, "numerator_as_long"):
                return float(s.numerator_as_long()) / float(s.denominator_as_long())
            return float(s.as_fraction())

        def _comp_to_drl(comp) -> str:
            idx = int(str(comp.arg(0))[1:]) - 1
            val = _z3_num_to_float(comp.arg(1))
            if is_le(comp):
                return f"-y_{idx}>={-val}"
            return f"y_{idx}>={val}"

        def _atoms_of(node):
            """Flatten And(a, b) into [a, b]; leave single inequalities as [node]."""
            if is_and(node):
                return list(node.children())
            return [node]

        lines = ["ordering " + " ".join(f"y_{i}" for i in range(self.n_outputs))]

        for child in z3_constr.children():
            if is_or(child):
                disjuncts = [_atoms_of(d) for d in child.children()]
                for combo in cartesian_product(*disjuncts):
                    parts = [_comp_to_drl(c) for c in combo]
                    lines.append(" or ".join(parts))
            elif is_and(child):
                for c in child.children():
                    lines.append(_comp_to_drl(c))
            else:
                lines.append(_comp_to_drl(child))

        return "\n".join(lines)
