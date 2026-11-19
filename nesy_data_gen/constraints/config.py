"""Pydantic configuration models for constraints."""

from pydantic import BaseModel, ConfigDict, Field


class BaseConstraintConfig(BaseModel):
    """Base configuration for all constraint types.

    When use_input_variables is True, constraints operate over V = I ∪ O.
    When False, constraints operate over V = O only (inputs are external).
    """

    seed: int = Field(default=0, description="Random seed for reproducibility")
    c: int = Field(default=1, ge=1, description="Number of unique constraints")
    b: int = Field(default=1, ge=1, description="Number of regions per constraint")
    n_inputs: int = Field(default=0, ge=0, description="Number of input variables")
    n_outputs: int = Field(default=10, ge=1, description="Number of output variables")
    use_input_variables: bool = Field(
        default=False,
        description="If True, input variables are part of V (V = I ∪ O). "
        "If False, V = O only and inputs are not referenced by the constraint.",
    )
    is_real: bool = Field(
        default=True, description="True for real-valued variables, False for Boolean"
    )
    require_all_inputs_valid: bool = Field(
        default=True,
        description="If True (ForAll): ∀I ∃O constraint(I,O) -- every input has a satisfying output. "
        "If False (Exists): ∃I ∃O constraint(I,O) -- some input/output pair satisfies it. "
        "Only meaningful when use_input_variables=True.",
    )

    @property
    def n_vars(self) -> int:
        """Number of variables in the constraint's variable set V."""
        if self.use_input_variables:
            return self.n_inputs + self.n_outputs
        return self.n_outputs


class CNFConstraintConfig(BaseConstraintConfig):
    """Configuration for CNF constraints over Boolean V = I ∪ O."""

    model_config = ConfigDict(extra="forbid", strict=True)
    is_real: bool = Field(default=False, description="CNF variables are Boolean")
    r: float = Field(
        default=3.0,
        ge=0.1,
        description="Clause-to-variable ratio (n_clauses = n_vars * r)",
    )
    k: int = Field(default=3, ge=1, description="Number of literals per clause (k-CNF)")

    @property
    def n_clauses(self) -> int:
        return int(self.n_vars * self.r)


class SMTConstraintConfig(BaseConstraintConfig):
    """Shared configuration for all SMT constraint variants.

    All SMT constraints use a CNF backbone (controlled by r, k) over V = I ∪ O
    and replace each Boolean literal with a real-valued expression via a G function.
    Subclass configs add G-function-specific parameters.
    """

    r: float = Field(
        default=3.0,
        ge=0.1,
        description="Clause-to-variable ratio for CNF backbone (n_clauses = n_vars * r)",
    )
    k: int = Field(
        default=3, ge=1, description="Literals per clause in CNF backbone (k-CNF)"
    )

    @property
    def n_clauses(self) -> int:
        return int(self.n_vars * self.r)


class SMTBoundedIntervalConstraintConfig(SMTConstraintConfig):
    """Configuration for bounded-interval G function.

    Each Boolean literal x_i is replaced with c_min <= x_i <= c_max,
    where c_min and c_max are random constants in [0, 1].
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    min_gap: float = Field(
        default=0.1,
        ge=0.0,
        description="Minimum gap between lower and upper bounds",
    )


class SMTLinearInequalityConstraintConfig(SMTConstraintConfig):
    """Configuration for general linear inequality G function.

    Each Boolean literal is replaced with c_0 + c_1*v_j1 + ... + c_n*v_jn <= 0,
    where n_var variables are sampled from V without replacement and n_var+1
    integer coefficients are drawn from [-coef_abs_max, coef_abs_max].
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    coef_abs_max: int = Field(
        default=5,
        ge=1,
        description="Max absolute value for sampled integer coefficients",
    )
    n_terms_min: int = Field(
        default=1, ge=1, description="Min variables sampled from V per term"
    )
    n_terms_max: int = Field(
        default=5, ge=1, description="Max variables sampled from V per term"
    )
