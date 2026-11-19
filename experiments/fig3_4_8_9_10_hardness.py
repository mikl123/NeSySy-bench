"""Training sweep with MLflow logging (parallel).
In this experiment we train simple neural network with different constraints.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "examples"))

from experiment_setting import (  # noqa: E402
    CONSTRAINT_PARAMS,
    DS_PARAMS_FILTERING,
    DS_PARAMS_FIXING,
    MODEL_CONFIG,
    fixing_params,
)

from experiments.utils.dataset_generation import generate_quality_dataset  # noqa: E402
from experiments.utils.io import load_dataset, save_dataset  # noqa: E402
from experiments.utils.mlflow_logger import MLflowLogger  # noqa: E402
from experiments.utils.run_args import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    add_run_args,
    apply_subset,
)
from experiments.utils.runner import run_sweep  # noqa: E402
from nesy_approaches.methods import ALL_METHODS, evaluate  # noqa: E402

ds_params = copy.deepcopy(DS_PARAMS_FIXING)
train_params = copy.deepcopy(MODEL_CONFIG)

# The paper's CNF datasets were generated with a generator learning rate of 0.01
CNF_DS_PARAMS = fixing_params(lr=0.01)

PRESETS = {
    # vary CVR
    "bi_cvr": {
        "kind": "boundary",
        "cvr": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        "seeds": [1, 2, 3, 4, 5],
        "use_input_variables": False,
        "c": [1],
        "hidden_layers": [(64, 3)],
        "learning_seeds": [1, 2, 3, 4, 5],
    },
    "li_cvr": {
        "kind": "linear_inequality",
        "cvr": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        "seeds": [1, 2, 3, 4, 5],
        "use_input_variables": False,
        "c": [1],
        "hidden_layers": [(64, 3)],
        "learning_seeds": [1, 2, 3, 4, 5],
    },
    "cnf_cvr": {
        "kind": "cnf",
        "ds_params": CNF_DS_PARAMS,
        "cvr": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        "seeds": [1, 2, 3, 4, 5],
        "use_input_variables": False,
        "c": [1],
        "hidden_layers": [(64, 3)],
        "learning_seeds": [1, 2, 3, 4, 5],
    },
    # V = O and I and vary CVR
    "bi_io_cvr": {
        "kind": "boundary",
        "cvr": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        "seeds": [1, 2, 3, 4, 5],
        "use_input_variables": True,
        "c": [1],
        "hidden_layers": [(64, 3)],
        "learning_seeds": [1, 2, 3, 4, 5],
    },
    "li_io_cvr": {
        "kind": "linear_inequality",
        "cvr": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        "seeds": [1, 2, 3, 4, 5],
        "use_input_variables": True,
        "c": [1],
        "hidden_layers": [(64, 3)],
        "learning_seeds": [1, 2, 3, 4, 5],
    },
    "cnf_io_cvr": {
        "kind": "cnf",
        "ds_params": CNF_DS_PARAMS,
        "cvr": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        "seeds": [1, 2, 3, 4, 5],
        "use_input_variables": True,
        "c": [1],
        "hidden_layers": [(64, 3)],
        "learning_seeds": [1, 2, 3, 4, 5],
    },
    # Fix CVR = 3 and change number of regions
    "cnf_regions": {
        "kind": "cnf",
        "ds_params": CNF_DS_PARAMS,
        "cvr": [3],
        "seeds": [1, 2, 3, 4, 5],
        "use_input_variables": False,
        "c": [1, 2, 3, 4, 5],
        "hidden_layers": [(64, 3)],
        "learning_seeds": [1, 2, 3, 4, 5],
    },
    "bi_regions": {
        "kind": "boundary",
        "cvr": [3],
        "seeds": [1, 2, 3, 4, 5],
        "use_input_variables": False,
        "c": [1, 2, 3, 4, 5],
        "hidden_layers": [(64, 3)],
        "learning_seeds": [1, 2, 3, 4, 5],
    },
    "li_regions": {
        "kind": "linear_inequality",
        "cvr": [3],
        "seeds": [1, 2, 3, 4, 5],
        "use_input_variables": False,
        "c": [1, 2, 3, 4, 5],
        "hidden_layers": [(64, 3)],
        "learning_seeds": [1, 2, 3, 4, 5],
    },
    # Vary term complexity (number of classes per inequality)
    "li_n_terms": {
        "kind": "linear_inequality",
        # the paper's term-complexity runs (Fig. 10) used 6 outputs, the other
        # linear-inequality experiments 10
        "constraint": {"n_outputs": 6},
        "cvr": [3],
        "seeds": [1, 2, 3, 4, 5],
        "use_input_variables": False,
        "c": [1],
        "n_terms": [2, 3, 4, 5, 6],
        "hidden_layers": [(64, 3)],
        "learning_seeds": [1, 2, 3, 4, 5],
    },
}


def _constraint_params(preset_name: str, kind: str) -> dict | None:
    """CONSTRAINT_PARAMS with the preset's constraint overrides (None: no overrides)."""
    overrides = PRESETS.get(preset_name, {}).get("constraint")
    if not overrides:
        return None
    return {kind: {**CONSTRAINT_PARAMS[kind], **overrides}}


def _dataset_dir(
    results_dir: Path,
    preset_name: str,
    cvr: float,
    seed: int,
    n_constraints: int,
    use_input_variables: bool,
    n_terms: int | None = None,
) -> Path:
    nt_suffix = f"_nt{n_terms}" if n_terms is not None else ""
    return (
        results_dir
        / preset_name
        / f"c{cvr}_s{seed}_nc{n_constraints}_uiv_{use_input_variables}{nt_suffix}"
    )


def _generate_and_save(
    cvr: float | int,
    seed: int,
    kind: str,
    use_input_variables: bool | None,
    n_constraints: int = 1,
    n_terms: int | None = None,
    preset_name: str = "",
    results_dir: Path = DEFAULT_RESULTS_DIR,
    device: str = "cpu",
) -> dict:
    """Generate one dataset per (cvr, seed) and save to disk."""
    (
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
        _,
    ) = generate_quality_dataset(
        kind=kind,
        seed=seed,
        cvr=cvr,
        ds_params=PRESETS.get(preset_name, {}).get("ds_params", ds_params),
        use_input_variables=use_input_variables,
        n_constraints=n_constraints,
        n_terms=n_terms,
        device=device,
        constraint_params=_constraint_params(preset_name, kind),
    )

    result_dir = _dataset_dir(
        results_dir, preset_name, cvr, seed, n_constraints, use_input_variables, n_terms
    )
    save_dataset(
        result_dir,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
        constraint,
        constr_params,
        current_ds_params,
        dataset_info,
    )

    return {
        "sub_experiment": f"dataset_c{cvr}_s{seed}_nc{n_constraints}_uiv_{use_input_variables}_nt{n_terms}",
        "metrics": {},
        "params": {
            "cvr": cvr,
            "kind": kind,
            "seed": seed,
            "n_constraints": n_constraints,
            "n_train": len(x_train),
            "n_valid": len(x_valid),
            "n_test": len(x_test),
            "dataset_info": dataset_info,
            **current_ds_params,
        },
        "tags": {"phase": "dataset_generation"},
    }


# ---------------------------------------------------------------------------
# Phase 2: run models on cached datasets
# ---------------------------------------------------------------------------


def _run_models(
    cvr: float | int,
    seed: int,
    kind: str,
    use_input_variables: bool | None,
    n_constraints: int,
    n_terms: int | None,
    preset_name: str,
    hidden_dim: int = 64,
    num_layers: int = 3,
    ls: int | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    device: str = "cpu",
) -> dict:
    """Run plain_nn for one (cvr, seed, hidden_config, learning_seed). Loads dataset from disk."""
    result_dir = _dataset_dir(
        results_dir, preset_name, cvr, seed, n_constraints, use_input_variables, n_terms
    )
    (
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
        constraint,
        constr_params,
        current_ds_params,
        dataset_info,
    ) = load_dataset(result_dir)

    override = {
        **train_params,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "device": device,
    }
    run_config = {**override, "learning_seed": ls} if ls is not None else override

    predicted, _ = ALL_METHODS["plain_nn"](
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        constraint,
        run_config,
    )
    metrics = evaluate(predicted, y_test, x_test, constraint)

    flat_metrics = MLflowLogger._flatten_dict(metrics)
    logged_metrics = {
        k: float(v)
        for k, v in flat_metrics.items()
        if isinstance(v, (int, float, np.floating))
    }
    ls_tag = f"_ls{ls}" if ls is not None else ""
    display_name = f"plain_nn_h{hidden_dim}_l{num_layers}"
    return {
        "sub_experiment": f"cvr_{cvr}_seed_{seed}_nc{n_constraints}_uiv_{use_input_variables}_nt{n_terms}{ls_tag}_{display_name}",
        "metrics": logged_metrics,
        "params": {
            "cvr": cvr,
            "kind": kind,
            "seed": seed,
            "learning_seed": ls,
            "model_type": display_name,
            "n_constraints": n_constraints,
            "constraint": constr_params,
            **current_ds_params,
            **run_config,
            "dataset_info": dataset_info,
            "n_train": len(x_train),
            "n_valid": len(x_valid),
            "n_test": len(x_test),
            "result_dir": str(result_dir),
        },
    }


def run_experiment(preset_name: str, args: argparse.Namespace) -> None:
    preset = apply_subset(PRESETS[preset_name], args)
    results_dir = args.results_dir.resolve()
    device = args.device
    kind = preset["kind"]
    cvr_values = preset["cvr"]
    seeds = preset["seeds"]
    use_input_variables = preset["use_input_variables"]
    c_values = preset.get("c", [1])
    n_terms_values = preset.get("n_terms", [None])
    learning_seeds = preset.get("learning_seeds", [None])

    # Phase 1: generate and cache datasets
    print(f"=== Phase 1: generating datasets for {preset_name} ===")
    dataset_tasks = [
        (c_val, s, kind, use_input_variables, nc, nt, preset_name, results_dir, device)
        for c_val in cvr_values
        for s in seeds
        for nc in c_values
        for nt in n_terms_values
    ]
    run_sweep(
        experiment_name=f"sup_train_{preset_name}_datasets",
        tasks=dataset_tasks,
        run_fn=_generate_and_save,
        tags={"constraint": kind, "preset": preset_name, "phase": "dataset"},
        max_workers=args.workers,
    )

    # Phase 2: run models on cached datasets
    print(f"=== Phase 2: running models for {preset_name} ===")
    hidden_layers = preset.get("hidden_layers", [(64, 3)])
    model_tasks = [
        (
            c_val,
            s,
            kind,
            use_input_variables,
            nc,
            nt,
            preset_name,
            hd,
            nl,
            ls,
            results_dir,
            device,
        )
        for c_val in cvr_values
        for s in seeds
        for nc in c_values
        for nt in n_terms_values
        for hd, nl in hidden_layers
        for ls in learning_seeds
    ]
    run_sweep(
        experiment_name=f"sup_train_{preset_name}",
        tasks=model_tasks,
        run_fn=_run_models,
        tags={"constraint": kind, "preset": preset_name, "phase": "models"},
        max_workers=args.workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Training sweep with MLflow.")
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="cnf_cvr",
        help="Preset name to run (default: cnf_cvr)",
    )
    add_run_args(parser, ["seeds", "learning_seeds", "cvr", "c", "n_terms"])
    args = parser.parse_args()
    run_experiment(preset_name=args.preset, args=args)


if __name__ == "__main__":
    main()
