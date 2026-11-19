"""Shared configuration for CNF (3-SAT) constraint examples."""

CNF_PARAMS = {
    # Common params
    "n_outputs": 15,  # Number of output variables (boolean)
    "n_inputs": 10,  # Number of input features (not part of the CNF)
    "c": 1,  # Number of unique constraint formulas
    "b": 1,  # Regions per constraint
    "seed": 41,
    # CNF constraint params
    "k": 3,  # Literals per clause (3-SAT)
    "r": 1,  # Clause ratio (n_clauses = n_outputs * r)
    "use_input_variables": False,
}

CNF_DS_PARAMS = {
    "generator_type": "filtering",
    "ds_params": {
        "n_inputs": CNF_PARAMS["n_inputs"],
        "n_outputs": CNF_PARAMS["n_outputs"],
        "n_samples": 10000,
        "exploration_set_size": 1000,
        "epochs": 10000,
        "lr": 0.001,
        "batch_size": 128,
        "hidden_dim": 64,
        "num_hidden": 3,
        "noise_std": 0.1,
        "max_iterations": 100,
        "losses": [
            {"name": "soft_repulsion", "weight": 1.0},
            {"name": "constraint", "weight": 1.0, "params": {"recalc_start": 20}},
        ],
    },
    "seed": 44,
    "test_size": 0.2,
    "valid_size": 0.1,
}

CNF_TRAIN_PARAMS = {
    "model": {
        "batch_size": 64,
        "epochs": 1000,
        "lr": 0.001,
        "hidden_dim": 64,
        "num_layers": 2,
        "patience": 20,
        "device": "cpu",
        "use_sigmoid": True,  # sigmoid kept for binary outputs + BCELoss
        "is_real": False,  # propositional mode → BCELoss
    },
}
