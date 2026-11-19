"""Shared configuration for SMT Linear Inequality constraint examples."""

LINEAR_INEQUALITY_PARAMS = {
    # Common params
    "n_outputs": 10,  # Number of output variables (continuous)
    "n_inputs": 4,  # Number of input features
    "c": 1,  # Number of unique constraint formulas
    "b": 1,  # Regions per constraint (total regions = c * b)
    "seed": 42,  # Random seed for reproducibility
    # SMT Linear Inequality constraint params
    "r": 1.0,  # Clause ratio (n_clauses = n_vars * r)
    "require_all_inputs_valid": True,  # Use True for ForAll, False for Exists
    "coef_abs_max": 5,  # Signed coefficients sampled from [-coef_abs_max, coef_abs_max]
    "n_terms_min": 2,  # Min variables sampled from V per term
    "n_terms_max": 3,  # Max variables sampled from V per term
}

LINEAR_INEQUALITY_DS_PARAMS = {
    "generator_type": "filtering",
    "ds_params": {
        "n_inputs": LINEAR_INEQUALITY_PARAMS["n_inputs"],
        "n_outputs": LINEAR_INEQUALITY_PARAMS["n_outputs"],
        "n_samples": 1000,
        "exploration_set_size": 10000,
        "epochs": 1000,
        "lr": 0.001,
    },
    "seed": 42,
    "test_size": 0.2,
    "valid_size": 0.1,
}

LINEAR_INEQUALITY_TRAIN_PARAMS = {
    "model": {
        "batch_size": 64,
        "epochs": 50,
        "lr": 1e-3,
        "hidden_dim": 64,
        "num_layers": 2,
        "patience": 20,
        "device": "cpu",
        "is_real": True,  # real-valued outputs → L1Loss
    },
}
