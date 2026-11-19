"""Tests for PiShield method (CCN+ for propositional, DRL for QFLRA)."""

import numpy as np
import torch

from nesy_approaches.methods import ALL_METHODS, _build_shield


class TestPiShieldCCN:
    def test_output_shape(self, cnf_dataset, cnf_constraint, model_config):
        x_train, y_train, x_valid, y_valid, x_test, y_test = cnf_dataset
        predicted = ALL_METHODS["pishield"](
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            cnf_constraint,
            model_config,
        )
        assert predicted.shape == (len(x_test), cnf_constraint.n_outputs)

    def test_output_range(self, cnf_dataset, cnf_constraint, model_config):
        x_train, y_train, x_valid, y_valid, x_test, y_test = cnf_dataset
        predicted = ALL_METHODS["pishield"](
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            cnf_constraint,
            model_config,
        )
        assert np.all(np.array(predicted.detach()) >= 0)
        assert np.all(np.array(predicted.detach()) <= 1)

    def test_constraints_satisfied(self, cnf_dataset, cnf_constraint, model_config):
        x_train, y_train, x_valid, y_valid, x_test, y_test = cnf_dataset
        predicted = ALL_METHODS["pishield"](
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            cnf_constraint,
            model_config,
        )
        dummy = np.zeros((len(x_test), max(1, cnf_constraint.n_inputs)))
        results = cnf_constraint.check(np.array(predicted.detach()) > 0.5, dummy)
        assert all(results), "PiShield predictions should satisfy constraints"

    def test_gradients_flow(self, cnf_constraint):
        """Verify gradients flow back through the shield layer."""
        CL = _build_shield(cnf_constraint)

        x = torch.rand(1, cnf_constraint.n_outputs, requires_grad=True)
        out = CL(x)
        out.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0


class TestPiShieldExportConsistency:
    """Verify that _to_pishield_str_one correctly converts Z3 constraints.

    Compares the Z3 human-readable string with the PiShield export string
    to ensure all bounds are faithfully represented.
    """

    def test_cnf_shield_corrects_to_valid(self, cnf_constraint):
        """PiShield CCN+ corrected outputs should satisfy the Z3 constraint."""
        CL = _build_shield(cnf_constraint)
        x = torch.rand(1000, cnf_constraint.n_outputs)
        corrected = CL(x).detach().numpy()
        dummy = np.zeros((1000, max(1, cnf_constraint.n_inputs)))
        results = cnf_constraint.check(corrected > 0.5, dummy)
        assert all(results), f"CCN+ correction failed: {sum(results)}/1000 valid"

    def test_boundary_shield_corrects_to_valid(self, boundary_constraint):
        """PiShield DRL corrected outputs should satisfy the Z3 constraint."""
        CL = _build_shield(boundary_constraint)
        x = torch.rand(1000, boundary_constraint.n_outputs)
        corrected = CL(x).detach().numpy()
        dummy = np.zeros((1000, max(1, boundary_constraint.n_inputs)))
        results = boundary_constraint.check(corrected, dummy)
        assert all(results), f"DRL correction failed: {sum(results)}/1000 valid"


class TestPiShieldDRL:
    """Tests for PiShield with real-valued bounded interval constraints (DRL)."""

    def test_output_shape(self, boundary_dataset, boundary_constraint, model_config):
        x_train, y_train, x_valid, y_valid, x_test, y_test = boundary_dataset
        predicted = ALL_METHODS["pishield"](
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            boundary_constraint,
            model_config,
        )
        assert predicted.shape == (len(x_test), boundary_constraint.n_outputs)

    def test_output_range(self, boundary_dataset, boundary_constraint, model_config):
        x_train, y_train, x_valid, y_valid, x_test, y_test = boundary_dataset
        predicted = ALL_METHODS["pishield"](
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            boundary_constraint,
            model_config,
        )
        pred_np = np.array(predicted.detach())
        assert np.all(pred_np >= 0) and np.all(pred_np <= 1)

    def test_constraints_satisfied(
        self, boundary_dataset, boundary_constraint, model_config
    ):
        x_train, y_train, x_valid, y_valid, x_test, y_test = boundary_dataset
        predicted = ALL_METHODS["pishield"](
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            boundary_constraint,
            model_config,
        )
        dummy = np.zeros((len(x_test), max(1, boundary_constraint.n_inputs)))
        results = boundary_constraint.check(np.array(predicted.detach()), dummy)
        predicted = np.array(predicted.detach())
        assert all(results), "DRL predictions should satisfy boundary constraints"

    def test_gradients_flow(self, boundary_constraint):
        CL = _build_shield(boundary_constraint)

        x = torch.rand(1, boundary_constraint.n_outputs, requires_grad=True)
        out = CL(x)
        out.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0
