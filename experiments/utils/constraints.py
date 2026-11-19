"""Shared constraint factory for experiments."""

from __future__ import annotations

import copy
from typing import Any

from experiment_setting import CONSTRAINT_PARAMS

from nesy_data_gen.constraints import (
    CNFConstraint,
    SMTBoundedIntervalConstraint,
    SMTLinearInequalityConstraint,
)


def create_constraint(
    kind: str,
    seed: int,
    cvr: float | int,
    use_input_variables: bool | None = None,
    n_constraints: int = 1,
    b: int | None = None,
    region_mapping: list[int] | None = None,
    constraint_params: dict[str, Any] | None = None,
    n_terms: int | None = None,
):
    """Build and generate a constraint from a common set of parameters."""
    params_source = (
        constraint_params if constraint_params is not None else CONSTRAINT_PARAMS
    )
    params = copy.deepcopy(params_source[kind])
    params["c"] = n_constraints
    params["seed"] = seed
    params["r"] = float(cvr)
    if b is not None:
        params["b"] = b
    if use_input_variables is not None:
        params["use_input_variables"] = use_input_variables
    if n_terms is not None:
        params["n_terms_min"] = n_terms
        params["n_terms_max"] = n_terms

    if kind == "boundary":
        constraint = SMTBoundedIntervalConstraint(**params)
    elif kind == "linear_inequality":
        constraint = SMTLinearInequalityConstraint(**params)
    elif kind == "cnf":
        constraint = CNFConstraint(**params)
    else:
        raise ValueError(f"Unknown constraint kind: {kind}")

    constraint.generate(region_mapping=region_mapping)
    return constraint, params
