"""In this experiment we compare the performance of plain NN with PiShield layer.
Namely DRL and CCN+ layers.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch
from experiment_setting import (  # noqa: E402
    DS_PARAMS_FIXING,
    MODEL_CONFIG,
    fixing_params,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "examples"))

from experiments.utils.dataset_generation import generate_quality_dataset  # noqa: E402
from experiments.utils.io import load_dataset, save_dataset  # noqa: E402
from experiments.utils.mlflow_logger import MLflowLogger
from experiments.utils.run_args import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    add_run_args,
    apply_subset,
)
from experiments.utils.runner import run_sweep  # noqa: E402
from nesy_approaches.methods import ALL_METHODS, evaluate  # noqa: E402

ds_params = copy.deepcopy(DS_PARAMS_FIXING)
model_config = copy.deepcopy(MODEL_CONFIG)

PRESETS = {
    "bi_methods": {
        "kind": "boundary",
        "ds_params": fixing_params(lr=0.01),
        "cvr": [1, 2, 3, 4],
        "seeds": [10, 20, 30, 40, 50],
        "n_samples": [5000],
        "methods": ["plain_nn", "plain_nn_postproc", "pishield"],
        "hidden_layers": [(128, 3), (512, 3), (1024, 4)],
        "learning_seeds": [1, 2, 3],
    },
    "cnf_methods": {
        "kind": "cnf",
        "ds_params": fixing_params(lr=0.01),
        "cvr": [1, 2, 3, 4],
        "seeds": [10, 20, 30, 40, 50],
        "n_samples": [5000],
        "methods": [
            "plain_nn",
            "plain_nn_postproc",
            "pishield",
            "sequential_beam",
        ],
        "hidden_layers": [(128, 3), (512, 3), (1024, 4)],
        "learning_seeds": [1, 2, 3],
    },
    "cnf_train_size": {
        "kind": "cnf",
        "ds_params": fixing_params(epochs=10000),
        "cvr": [2],
        "seeds": [10, 20, 30, 40, 50],
        "n_samples": [100, 500, 5000, 10000, 20000],
        "methods": [
            "plain_nn",
            "plain_nn_postproc",
            "pishield",
            "sequential_beam",
        ],
        "hidden_layers": [(128, 3)],
        "learning_seeds": [1, 2, 3, 4, 5],
    },
}


def _dataset_dir(
    results_dir: Path, preset_name: str, cvr: float, seed: int, n: int
) -> Path:
    return results_dir / preset_name / f"c{cvr}_s{seed}_n{n}"


def _generate_and_save(
    cvr: float | int,
    seed: int,
    kind: str,
    n_samples_list: list[int],
    preset_name: str = "",
    full_n: int | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    device: str = "cpu",
) -> list[dict]:
    """Generate one dataset of *full_n* samples per (cvr, seed) and save the
    first n samples of it for every n in *n_samples_list*."""
    full_n = full_n or max(n_samples_list)
    full_ds_params = copy.deepcopy(PRESETS[preset_name].get("ds_params", ds_params))
    full_ds_params["ds_params"]["n_samples"] = full_n

    (
        x_train_full,
        y_train_full,
        x_valid_full,
        y_valid_full,
        x_test,
        y_test,
        constraint,
        constr_params,
        dataset_info,
        current_ds_params,
        _,
    ) = generate_quality_dataset(
        kind=kind, seed=seed, cvr=cvr, ds_params=full_ds_params, device=device
    )

    results = []
    for n in sorted(n_samples_list):
        frac = n / full_n
        n_train = max(1, round(len(x_train_full) * frac))
        n_valid = max(1, round(len(x_valid_full) * frac))
        x_train = x_train_full[:n_train]
        y_train = y_train_full[:n_train]
        x_valid = x_valid_full[:n_valid]
        y_valid = y_valid_full[:n_valid]

        result_dir = _dataset_dir(results_dir, preset_name, cvr, seed, n)
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

        results.append(
            {
                "sub_experiment": f"dataset_c{cvr}_s{seed}_n{n}",
                "metrics": {},
                "params": {
                    "cvr": cvr,
                    "kind": kind,
                    "seed": seed,
                    "n_train": n_train,
                    "n_valid": n_valid,
                    "n_test": len(x_test),
                    "dataset_info": dataset_info,
                    **current_ds_params,
                },
                "tags": {"phase": "dataset_generation"},
            }
        )

    return results


_load_cached_dataset = load_dataset


def _run_models(
    cvr: float | int,
    seed: int,
    kind: str,
    n_samples_list: list[int],
    preset_name: str,
    hidden_dim: int,
    num_layers: int,
    ls: int | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    device: str = "cpu",
) -> list[dict]:
    """Run all methods for one (cvr, seed, hidden_config, learning_seed). Loads dataset from disk."""
    preset = PRESETS[preset_name]
    methods = preset["methods"]
    results = []

    for n in sorted(n_samples_list):
        result_dir = _dataset_dir(results_dir, preset_name, cvr, seed, n)
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
        ) = _load_cached_dataset(result_dir)

        override = {
            **model_config,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "device": device,
        }
        run_config = {**override, "learning_seed": ls} if ls is not None else override

        # Train base NN once, reuse for plain_nn_postproc and sequential_*
        base_trainer = None
        _REUSE_METHODS = {
            "plain_nn_postproc",
            "sequential_beam",
            "sequential_greedy",
            "sequential_fix",
        }
        if "plain_nn" in methods or any(m in methods for m in _REUSE_METHODS):
            from nesy_approaches.methods import plain_nn as _plain_nn

            base_preds, base_trainer = _plain_nn(
                x_train,
                y_train,
                x_valid,
                y_valid,
                x_test,
                constraint,
                run_config,
            )

        for method_name in methods:
            # DRL is not run for bounded intervals with cvr > 2 (too slow), as in the paper
            if method_name == "pishield" and cvr > 2 and kind == "boundary":
                continue
            display_name = f"{method_name}_h{hidden_dim}_l{num_layers}"

            try:
                if method_name == "plain_nn":
                    predicted = base_preds
                elif method_name in _REUSE_METHODS:
                    method_fn = ALL_METHODS[method_name]
                    predicted = method_fn(
                        x_train,
                        y_train,
                        x_valid,
                        y_valid,
                        x_test,
                        constraint,
                        run_config,
                        base_trainer=base_trainer,
                    )
                else:
                    method_fn = ALL_METHODS[method_name]
                    predicted = method_fn(
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
                results.append(
                    {
                        "sub_experiment": f"{display_name}_c{cvr}_s{seed}_n{n}{ls_tag}",
                        "metrics": logged_metrics,
                        "params": {
                            "cvr": cvr,
                            "kind": kind,
                            "seed": seed,
                            "model_type": display_name,
                            "learning_seed": ls,
                            "constraint": constr_params,
                            **current_ds_params,
                            **run_config,
                            "dataset_info": dataset_info,
                            "n_train": len(x_train),
                            "n_valid": len(x_valid),
                            "n_test": len(x_test),
                            "result_dir": str(result_dir),
                        },
                        "tags": {"model_type": display_name},
                    }
                )
            except Exception as e:
                ls_tag = f"_ls{ls}" if ls is not None else ""
                print(
                    f"  [SKIP] {display_name} failed for cvr={cvr} seed={seed} n={n} ls={ls}: {e}"
                )
                results.append(
                    {
                        "sub_experiment": f"FAILED_{display_name}_c{cvr}_s{seed}_n{n}{ls_tag}",
                        "metrics": {},
                        "params": {
                            "cvr": cvr,
                            "kind": kind,
                            "seed": seed,
                            "model_type": display_name,
                            "learning_seed": ls,
                        },
                        "tags": {
                            "model_type": display_name,
                            "status": "failed",
                            "error": str(e)[:250],
                        },
                    }
                )

    return results


def run_experiment(preset_name: str, args: argparse.Namespace) -> None:
    # the full dataset size stays the preset's largest, also when --n-samples
    # selects only some of the sizes
    full_n = max(PRESETS[preset_name]["n_samples"])
    preset = apply_subset(PRESETS[preset_name], args)
    results_dir = args.results_dir.resolve()
    device = args.device
    kind = preset["kind"]
    hidden_layers = preset.get("hidden_layers", [(64, 3)])

    # # Phase 1: generate and cache datasets
    print(f"=== Phase 1: generating datasets for {preset_name} ===")
    dataset_tasks = [
        (c, s, kind, preset["n_samples"], preset_name, full_n, results_dir, device)
        for c in preset["cvr"]
        for s in preset["seeds"]
    ]
    run_sweep(
        experiment_name=f"{preset_name}_datasets",
        tasks=dataset_tasks,
        run_fn=_generate_and_save,
        tags={"constraint": kind, "preset": preset_name, "phase": "dataset"},
        max_workers=args.workers,
    )

    # Phase 2: run models — one task per (cvr, seed, hidden_config, learning_seed)
    print(f"=== Phase 2: running models for {preset_name} ===")
    learning_seeds = preset.get("learning_seeds", [None])
    model_tasks = [
        (
            c,
            s,
            kind,
            preset["n_samples"],
            preset_name,
            hd,
            nl,
            ls,
            results_dir,
            device,
        )
        for c in preset["cvr"]
        for s in preset["seeds"]
        for hd, nl in hidden_layers
        for ls in learning_seeds
    ]
    run_sweep(
        experiment_name=f"{preset_name}",
        tasks=model_tasks,
        run_fn=_run_models,
        tags={"constraint": kind, "preset": preset_name, "phase": "models"},
        max_workers=args.workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PiShield vs Plain NN complexity sweep."
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="cnf_methods",
        help="Preset name to run (default: cnf_methods)",
    )
    add_run_args(
        parser, ["seeds", "learning_seeds", "cvr", "hidden_layers", "n_samples"]
    )
    args = parser.parse_args()
    run_experiment(preset_name=args.preset, args=args)


if __name__ == "__main__":
    main()
