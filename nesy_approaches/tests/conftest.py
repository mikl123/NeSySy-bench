"""Shared fixtures for nesy_approaches tests."""

import numpy as np
import pytest
import torch

from nesy_data_gen.constraints import CNFConstraint, SMTBoundedIntervalConstraint


@pytest.fixture(scope="module")
def cnf_constraint():
    c = CNFConstraint(
        n_outputs=6,
        n_inputs=1,
        use_input_variables=False,
        c=1,
        b=1,
        seed=7,
        r=2.0,
    )
    c.generate()
    return c


@pytest.fixture(scope="module")
def cnf_dataset(cnf_constraint):
    """Small synthetic dataset for CNF constraint."""
    np.random.seed(42)
    n_train, n_valid, n_test = 50, 10, 20
    n_in, n_out = 4, cnf_constraint.n_outputs

    x_train = torch.randn(n_train, n_in)
    x_valid = torch.randn(n_valid, n_in)
    x_test = torch.randn(n_test, n_in)

    def gen_valid_labels(n, constraint):
        labels = []
        dummy_input = np.zeros((1, max(1, constraint.n_inputs)))
        while len(labels) < n:
            candidate = np.random.randint(0, 2, size=(1, n_out)).astype(float)
            if constraint.check(candidate, dummy_input)[0]:
                labels.append(candidate[0])
        return torch.tensor(np.array(labels), dtype=torch.float32)

    y_train = gen_valid_labels(n_train, cnf_constraint)
    y_valid = gen_valid_labels(n_valid, cnf_constraint)
    y_test = gen_valid_labels(n_test, cnf_constraint)

    return x_train, y_train, x_valid, y_valid, x_test, y_test


@pytest.fixture(scope="module")
def boundary_constraint():
    c = SMTBoundedIntervalConstraint(
        n_outputs=4,
        n_inputs=2,
        use_input_variables=False,
        c=1,
        b=1,
        seed=42,
        r=1.0,
        min_gap=0.1,
    )
    c.generate()
    return c


@pytest.fixture(scope="module")
def boundary_dataset(boundary_constraint):
    """Small synthetic dataset for bounded interval constraint."""
    np.random.seed(42)
    n_train, n_valid, n_test = 50, 10, 20
    n_in = 2
    n_out = boundary_constraint.n_outputs

    x_train = torch.randn(n_train, n_in)
    x_valid = torch.randn(n_valid, n_in)
    x_test = torch.randn(n_test, n_in)

    # Generate random real labels in [0, 1] that satisfy the constraint
    def gen_valid_labels(n, constraint):
        labels = []
        dummy_input = np.zeros((1, max(1, constraint.n_inputs)))
        while len(labels) < n:
            candidate = np.random.rand(1, n_out).astype(np.float32)
            if constraint.check(candidate, dummy_input)[0]:
                labels.append(candidate[0])
        return torch.tensor(np.array(labels), dtype=torch.float32)

    y_train = gen_valid_labels(n_train, boundary_constraint)
    y_valid = gen_valid_labels(n_valid, boundary_constraint)
    y_test = gen_valid_labels(n_test, boundary_constraint)

    return x_train, y_train, x_valid, y_valid, x_test, y_test


@pytest.fixture(scope="module")
def model_config():
    return {
        "batch_size": 32,
        "epochs": 50,
        "lr": 0.01,
        "hidden_dim": 16,
        "num_layers": 2,
        "patience": 10,
        "device": "cpu",
        "seq_hidden_size": 16,
        "seq_num_layers": 3,
        "seq_dropout": 0.0,
    }
