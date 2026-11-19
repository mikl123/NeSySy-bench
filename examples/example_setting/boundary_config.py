"""Shared configuration for SMT Bounded-Interval constraint examples."""

BOUNDARY_PARAMS = {
    # Common params
    "n_outputs": 8,  # Number of output variables (continuous)
    "n_inputs": 10,  # Number of input features
    "c": 1,  # Number of unique constraint formulas
    "b": 1,  # Regions per constraint (total regions = c * b)
    "seed": 42,  # Random seed for reproducibility
    # SMT Bounded-Interval constraint params
    "r": 2,  # Clause ratio (n_clauses = n_vars * r)
    "min_gap": 0.2,  # Minimum gap between interval bounds
}

BOUNDARY_DS_PARAMS = {
    "generator_type": "fixing",
    "ds_params": {
        "n_inputs": BOUNDARY_PARAMS["n_inputs"],
        "n_outputs": BOUNDARY_PARAMS["n_outputs"],
        "n_samples": 3000,
        "exploration_set_size": 3000,
        "epochs": 1000,
        "lr": 0.01,
        "batch_size": 128,
        "fix_noise_std": 0.3,
        "fix_iterations": 1,
        "losses": [
            {"name": "soft_repulsion", "weight": 1.0},
            {"name": "constraint", "weight": 10.0, "params": {"recalc_start": 20}},
        ],
    },
    "seed": 42,
    "test_size": 0.2,
    "valid_size": 0.1,
}

BOUNDARY_TRAIN_PARAMS = {
    "model": {
        "batch_size": 64,
        "epochs": 100,
        "lr": 1e-3,
        "hidden_dim": 64,
        "num_layers": 2,
        "patience": 20,
        "device": "cpu",
        "is_real": True,  # real-valued outputs → L1Loss
    },
}
