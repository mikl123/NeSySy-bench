"""Shared experiment defaults.
Each experiment imports these and overrides only what differs
"""

import copy

DS_PARAMS_FIXING = {
    "generator_type": "fixing",
    "ds_params": {
        "n_inputs": 10,
        "n_outputs": 10,  # overridden per constraint type (CNF = 10, SMT = 6)
        "n_samples": 3000,  # overridden in some experiments
        "hidden_dim": 128,
        "num_hidden": 5,
        "epochs": 1000,
        "lr": 0.001,
        "exploration_set_size": 3000,
        "batch_size": 128,
        "weight_decay": 1e-4,
        "losses": [
            {"name": "soft_repulsion", "weight": 1.0},
            {"name": "constraint", "weight": 1.0, "params": {"recalc_start": 20}},
        ],
        # Fixing specific settings
        "fix_noise_std": 0.1,
        "fix_iterations": 10,
    },
    "test_size": 0.2,
    "valid_size": 0.1,
}

DS_PARAMS_FILTERING = {
    "generator_type": "filtering",
    "ds_params": {
        "n_inputs": 10,
        "n_outputs": 10,  # overridden per constraint type (CNF = 10, SMT = 6)
        "n_samples": 3000,  # overridden in some experiments
        "hidden_dim": 128,
        "num_hidden": 5,
        "epochs": 1000,
        "lr": 0.01,
        "exploration_set_size": 3000,
        "batch_size": 128,
        "weight_decay": 1e-4,
        "losses": [
            {"name": "soft_repulsion", "weight": 1.0},
            {"name": "constraint", "weight": 1.0, "params": {"recalc_start": 20}},
        ],
        # Filtering specific settings
        "noise_std": 0.1,
        "max_iterations": 100,
    },
    "test_size": 0.2,
    "valid_size": 0.1,
}


MODEL_CONFIG = {
    "batch_size": 128,
    "epochs": 5000,
    "lr": 0.001,
    "hidden_dim": 64,
    "num_layers": 3,
    "dropout": 0.1,
    "patience": 20,
    "device": "cpu",
    "learning_seed": 42,
}

CONSTRAINT_PARAMS = {
    "boundary": {
        "n_outputs": 10,
        "n_inputs": 10,
        "c": 1,  # overridden in some experiments
        "r": 1,  # overridden in some experiments
        "seed": 42,  # overridden in some experiments
        "b": 1,
        "require_all_inputs_valid": True,
        "use_input_variables": False,  # overridden in some experiments
        # BI specific settings
        "min_gap": 0.3,
    },
    "linear_inequality": {
        "n_outputs": 10,  # the term-complexity experiment (Fig. 10) uses 6
        "n_inputs": 10,
        "c": 1,  # overridden in some experiments
        "r": 1,  # overridden in some experiments
        "seed": 42,  # overridden in some experiments
        "b": 1,
        "require_all_inputs_valid": True,
        "use_input_variables": False,  # overridden in some experiments
        # Linear inequality specific metrics
        "coef_abs_max": 5,
        "n_terms_min": 3,
        "n_terms_max": 3,
    },
    "cnf": {
        "n_outputs": 15,
        "n_inputs": 10,
        "c": 1,  # overridden in some experiments
        "r": 1,  # overridden in some experiments
        "seed": 42,  # overridden in some experiments
        "k": 3,
        "b": 1,
        "require_all_inputs_valid": True,
        "use_input_variables": False,  # overridden in some experiments
    },
}


def fixing_params(**overrides) -> dict:
    """Copy of DS_PARAMS_FIXING with the given generator-mapping settings overridden."""
    params = copy.deepcopy(DS_PARAMS_FIXING)
    params["ds_params"].update(overrides)
    return params
