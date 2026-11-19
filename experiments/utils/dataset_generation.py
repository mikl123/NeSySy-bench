"""Shared dataset generation with quality retry logic."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from experiments.utils.constraints import create_constraint
from nesy_approaches.utils import generate_dataset
from nesy_data_gen.generators.base import NoSatisfyingSamplesError

MAX_RETRIES = 20


def generate_quality_dataset(
    kind: str,
    seed: int,
    cvr: float | int,
    ds_params: dict,
    use_input_variables: bool | None = None,
    n_constraints: int = 1,
    n_terms: int | None = None,
    b: int | None = None,
    region_mapping: list[int] | None = None,
    sup_regions: list[int] | None = None,
    max_retries: int = MAX_RETRIES,
    device="cpu",
    constraint_params: dict | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Any,
    dict,
    dict,
]:
    """Generate constraint + dataset, retrying with different seeds if low quality or no valid samples.

    Returns:
        x_train, y_train, x_valid, y_valid, x_test, y_test,
        constraint, constr_params, dataset_info, current_ds_params,
        x_unsup (None unless sup_regions is set)
    """
    dataset_info = None
    x_unsup = None
    for attempt in range(max_retries):
        attempt_seed = seed + attempt * 1000
        constraint, constr_params = create_constraint(
            kind,
            attempt_seed,
            cvr,
            use_input_variables=use_input_variables,
            n_constraints=n_constraints,
            n_terms=n_terms,
            b=b,
            region_mapping=region_mapping,
            constraint_params=constraint_params,
        )

        local_ds_params = copy.deepcopy(ds_params)
        local_ds_params["ds_params"]["n_outputs"] = constraint.n_outputs
        current_ds_params = {**local_ds_params, "seed": attempt_seed}

        try:
            (
                x_train,
                y_train,
                x_valid,
                y_valid,
                x_test,
                y_test,
                x_unsup,
                dataset_info,
            ) = generate_dataset(
                current_ds_params, constraint, sup_regions=sup_regions, device=device
            )
        except NoSatisfyingSamplesError as e:
            print(
                f"  No valid samples for cvr={cvr} seed={seed} attempt={attempt + 1}: {e}, retrying..."
            )
            continue

        n_target = local_ds_params["ds_params"].get("n_samples", 0)
        n_generated = dataset_info.get("n_samples_generated", 0)
        is_low_quality = dataset_info.get("low_quality", False)
        is_deficit = n_generated < n_target
        if not is_low_quality and not is_deficit:
            dataset_info["generation_attempts"] = attempt + 1
            if attempt > 0:
                print(
                    f"  Good quality dataset on attempt {attempt + 1} (seed={attempt_seed})"
                )
            break

        reasons = []
        if is_low_quality:
            reasons.append("low_quality")
        if is_deficit:
            reasons.append(f"deficit ({n_generated}/{n_target})")
        print(
            f"  Bad dataset for cvr={cvr} seed={seed} attempt={attempt + 1}: {', '.join(reasons)}, retrying..."
        )
    else:
        if dataset_info is None:
            raise NoSatisfyingSamplesError(
                f"All {max_retries} attempts failed to generate any valid samples "
                f"for cvr={cvr} seed={seed}"
            )
        print(
            f"  WARNING: All {max_retries} attempts produced low quality for cvr={cvr} seed={seed}"
        )
        dataset_info["low_quality_all_attempts"] = True
        dataset_info["generation_attempts"] = max_retries

    return (
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
        constraint,
        constr_params,
        dataset_info,
        current_ds_params,
        x_unsup,
    )
