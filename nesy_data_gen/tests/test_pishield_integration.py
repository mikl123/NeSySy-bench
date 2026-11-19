"""Tests for PiShield integration: gradient flow and constraint satisfaction."""

import os
import tempfile

import numpy as np
import pytest
import torch
from pishield.shield_layer import build_shield_layer

from nesy_data_gen.constraints import CNFConstraint, SMTBoundedIntervalConstraint


def _write_tmp(text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


@pytest.fixture(scope="module")
def cnf():
    c = CNFConstraint(
        n_outputs=6, n_inputs=0, use_input_variables=False, c=1, b=1, seed=7, r=2.0
    )
    c.generate()
    return c


@pytest.fixture(scope="module")
def smt_bi():
    s = SMTBoundedIntervalConstraint(
        n_outputs=4,
        n_inputs=0,
        use_input_variables=False,
        c=1,
        b=1,
        seed=42,
        r=1.0,
        min_gap=0.1,
    )
    s.generate()
    return s


class TestPiShieldCNF:
    def test_gradients_flow(self, cnf):
        path = _write_tmp(cnf._to_pishield_str_one(0))
        try:
            layer = build_shield_layer(
                cnf.n_outputs,
                path,
                ordering_choice="degree",
                requirements_type="propositional",
            )
            x = torch.full((1, cnf.n_outputs), 0.5, requires_grad=True)
            loss = layer(x).sum()
            loss.backward()
            assert x.grad is not None and x.grad.abs().sum() > 0
        finally:
            os.unlink(path)

    def test_constraints_satisfied_uniform_input(self, cnf):
        path = _write_tmp(cnf._to_pishield_str_one(0))
        try:
            layer = build_shield_layer(
                cnf.n_outputs,
                path,
                ordering_choice="degree",
                requirements_type="propositional",
            )
            x = torch.rand((1000, cnf.n_outputs))
            out = layer(x).detach().numpy()
            dummy = np.zeros((1000, 1))
            assert all(cnf.check(out > 0.5, dummy))
        finally:
            os.unlink(path)


class TestPiShieldSMT:
    def test_gradients_flow(self, smt_bi):
        path = _write_tmp(smt_bi._to_pishield_str_one(0))
        try:
            layer = build_shield_layer(smt_bi.n_outputs, path)
            x = torch.full((1, smt_bi.n_outputs), 0.5, requires_grad=True)
            loss = layer(x).sum()
            loss.backward()
            assert x.grad is not None and x.grad.abs().sum() > 0
        finally:
            os.unlink(path)

    def test_constraints_satisfied_uniform_input(self, smt_bi):
        path = _write_tmp(smt_bi._to_pishield_str_one(0))
        try:
            layer = build_shield_layer(
                smt_bi.n_outputs, path, requirements_type="qflra"
            )
            x = torch.rand((1000, smt_bi.n_outputs))
            out = layer(x).detach().numpy()
            dummy = np.zeros((1000, 1))
            assert all(smt_bi.check(out, dummy))
        finally:
            os.unlink(path)
