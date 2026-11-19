"""SMT Linear Inequality constraint: general linear inequality G function."""

import random
from typing import Any

from z3 import Sum

from nesy_data_gen.constraints.config import SMTLinearInequalityConstraintConfig
from nesy_data_gen.constraints.smt import SMTConstraint


class SMTLinearInequalityConstraint(SMTConstraint, SMTLinearInequalityConstraintConfig):
    """Linear inequality G: c_0 + c_1*v_j1 + ... + c_n*v_jn <= 0.

    n_var variables are sampled from V = I ∪ O without replacement,
    with n_var+1 random integer coefficients from [-coef_abs_max, coef_abs_max].
    Uses CNF backbone to provide the clause structure.
    """

    def __str__(self) -> str:
        if self.use_input_variables:
            mode = "∀I∃O" if self.require_all_inputs_valid else "∃I∃O"
            vars_desc = f"{self.n_outputs}O+{self.n_inputs}I vars"
        else:
            mode = "V=O"
            vars_desc = f"{self.n_outputs}O vars"
        return (
            f"SMTLinearInequality({vars_desc}, "
            f"coef_max={self.coef_abs_max}, {mode}, {self.c}x{self.b} regions)"
        )

    def _sample_nonzero_coef(self) -> int:
        return random.choice(
            [c for c in range(-self.coef_abs_max, self.coef_abs_max + 1) if c != 0]
        )

    def _apply_g(
        self,
        lit: int,
        z3_var: Any,
        z3_out_vars: dict[int, Any],
        z3_inp_vars: dict[int, Any],
        max_attempts: int = 1000,
    ) -> Any:
        all_vars = list(z3_out_vars.values()) + list(z3_inp_vars.values())
        for _ in range(max_attempts):
            n_terms = random.randint(
                self.n_terms_min,
                min(self.n_terms_max, len(all_vars)),
            )
            sampled = random.sample(all_vars, k=n_terms)
            coefs = [self._sample_nonzero_coef() for _ in range(n_terms + 1)]
            if n_terms >= 2 and all(c > 0 for c in coefs):
                coefs[random.randrange(n_terms)] = -random.randint(1, self.coef_abs_max)
            elif n_terms >= 2 and all(c < 0 for c in coefs):
                coefs[random.randrange(n_terms)] = random.randint(1, self.coef_abs_max)
            constant = coefs[-1]
            var_coefs = coefs[:-1]

            # Check: for variables in [0,1], is the inequality non-trivial?
            # f(x) = constant + sum(ci * xi), xi in [0,1]
            # min(f) = constant + sum(ci for ci < 0)
            # max(f) = constant + sum(ci for ci > 0)
            f_min = constant + sum(c for c in var_coefs if c < 0)
            f_max = constant + sum(c for c in var_coefs if c > 0)

            if f_min > 0:
                continue  # always false — no assignment satisfies it
            if f_max <= 0:
                continue  # always true — every assignment satisfies it

            terms = [v * c for v, c in zip(sampled, var_coefs)]
            return constant + Sum(terms) <= 0

        return constant + Sum(terms) <= 0
