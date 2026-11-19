import copy
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "examples"))

import mlflow

from experiments.utils.dataset_generation import generate_quality_dataset
from experiments.utils.mlflow_logger import MLflowLogger
from experiments.utils.run_args import parse_distant_args
from nesy_approaches.methods import ALL_METHODS, evaluate
from nesy_data_gen.utils.deeplog_interface import constraint_to_deeplog_module

ARGS = parse_distant_args(
    "Figure 7 (left): DeepProbLog vs Base, varying the clause-to-variable ratio",
    "clause-to-variable ratios",
)
DEVICE = ARGS.device
EXPERIMENT_NAME = "test_5_distant5"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5005")


DS_PARAMS_FILTERING = {
    "generator_type": "filtering",
    "ds_params": {
        "n_inputs": 20,
        "n_outputs": 5,
        "n_samples": 5000,
        "hidden_dim": 1024,
        "num_hidden": 2,
        "epochs": 1000,
        "lr": 0.001,
        "exploration_set_size": 10000,
        "batch_size": 128,
        "losses": [
            {"name": "soft_repulsion", "weight": 1.0},
        ],
        "noise_std": 0.2,
        "max_iterations": 100,
    },
    "test_size": 0.2,
    "valid_size": 0.1,
}

TRAIN_PARAMS = {
    "batch_size": 128,
    "epochs": 5000,
    "lr": 0.001,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.1,
    "patience": 20,
    "device": DEVICE,
    "use_sigmoid": True,
    "is_real": False,
}

CNF_CONSTRAINT_PARAMS = {
    "n_outputs": DS_PARAMS_FILTERING["ds_params"]["n_outputs"],
    "n_inputs": DS_PARAMS_FILTERING["ds_params"]["n_inputs"],
    "c": 1,
    "r": 1,
    "seed": 42,
    "k": 3,  # With k=3 it is a 3-SAT problem
    "b": 1,
    "require_all_inputs_valid": True,
    "use_input_variables": False,
}

CVR_VALUES = [
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.2,
    2.4,
    2.6,
    2.8,
    3.0,
    3.2,
    3.4,
    3.6,
    3.8,
    4.0,
]
SEEDS = ARGS.seeds or [10, 20, 30, 40, 50]  # Seed to create dataset
LEARNING_SEEDS = ARGS.learning_seeds or [1, 2, 3, 4, 5]  # Seed for model
CVR_VALUES = ARGS.values or CVR_VALUES

logger = MLflowLogger(EXPERIMENT_NAME, tracking_uri=TRACKING_URI)

for seed in SEEDS:
    for cvr in CVR_VALUES:
        print(f"\n{'='*60}")
        print(f"CVR={cvr}, seed={seed}")
        print(f"{'='*60}")

        # Generate high-quality dataset with retries
        ds_params = copy.deepcopy(DS_PARAMS_FILTERING)
        ds_params["ds_params"]["distant_supervision"] = True

        (
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            y_test,
            cnf,
            constr_params,
            dataset_info,
            current_ds_params,
            _,
        ) = generate_quality_dataset(
            kind="cnf",
            seed=seed,
            cvr=cvr,
            ds_params=ds_params,
            device=DEVICE,
            constraint_params={"cnf": CNF_CONSTRAINT_PARAMS},
        )

        # Distant supervision: Y has n_outputs neural columns + 1 query column
        y_neural_train, y_query_train = y_train[:, :-1], y_train[:, -1:]
        y_neural_valid, y_query_valid = y_valid[:, :-1], y_valid[:, -1:]
        y_query_test = y_test[:, -1:]

        n_outputs = y_neural_train.shape[1]

        # Build constraint module for semantic loss
        constraint_module = constraint_to_deeplog_module(cnf._constraints[0], n_outputs)
        constraint_module.to(DEVICE)

        for ls in LEARNING_SEEDS:

            run_config = {**TRAIN_PARAMS, "learning_seed": ls}
            common_params = {
                "cvr": cvr,
                "seed": seed,
                "learning_seed": ls,
                **run_config,
            }

            # --- plain_nn ---
            print(f"  plain_nn (ls={ls})")
            predicted, trainer = ALL_METHODS["plain_nn"](
                x_train,
                y_query_train,
                x_valid,
                y_query_valid,
                x_test,
                cnf,
                run_config,
            )
            metrics = evaluate(predicted, y_query_test, x_test, cnf)
            flat = {
                k: float(v)
                for k, v in MLflowLogger._flatten_dict(metrics).items()
                if isinstance(v, (int, float, np.floating))
            }

            all_metrics = {
                **flat,
                "final_train_loss": trainer.history["train_loss"][-1],
                "final_val_loss": trainer.history["val_loss"][-1],
            }
            logger.start_run(
                f"plain_nn_c{cvr}_s{seed}_ls{ls}",
                {
                    **common_params,
                    "method": "plain_nn",
                    "dataset_info": dataset_info,
                },
            )
            logger.log_metrics(all_metrics)
            logger.log_tags({"model_type": "plain_nn"})
            # Save training info as artifact (too large for params)
            import tempfile

            training_info_path = os.path.join(
                tempfile.gettempdir(), "training_info.json"
            )
            with open(training_info_path, "w") as f:
                f.write(dataset_info.get("base_model_training", "{}"))
            mlflow.log_artifact(training_info_path)
            logger.end_run()

            # --- plain_nn_sl ---
            print(f"  plain_nn_sl (ls={ls})")
            predicted = ALL_METHODS["plain_nn_sl"](
                x_train,
                y_neural_train,
                y_query_train,
                x_valid,
                y_neural_valid,
                y_query_valid,
                x_test,
                constraint_module,
                run_config,
            )
            metrics = evaluate(predicted, y_query_test, x_test, cnf)
            flat = {
                k: float(v)
                for k, v in MLflowLogger._flatten_dict(metrics).items()
                if isinstance(v, (int, float, np.floating))
            }

            logger.start_run(
                f"plain_nn_sl_c{cvr}_s{seed}_ls{ls}",
                {**common_params, "method": "plain_nn_sl"},
            )
            logger.log_metrics(flat)
            logger.log_tags({"model_type": "plain_nn_sl"})
            logger.end_run()

print("\nDone.")
