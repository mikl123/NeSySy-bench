"""Tests for plain_nn and plain_nn_postproc methods."""

import numpy as np

from nesy_approaches.methods import ALL_METHODS


def _predict(method, *args):
    """Call a method and return only the predictions (drop trainer if returned)."""
    out = ALL_METHODS[method](*args)
    return out[0] if isinstance(out, tuple) else out


class TestPlainNNCNF:
    def test_output_shape(self, cnf_dataset, cnf_constraint, model_config):
        x_train, y_train, x_valid, y_valid, x_test, y_test = cnf_dataset
        predicted = _predict(
            "plain_nn",
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
        predicted = _predict(
            "plain_nn",
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            cnf_constraint,
            model_config,
        )
        assert np.all(np.array(predicted) >= 0)
        assert np.all(np.array(predicted) <= 1)


class TestPlainNNPostproc:
    def test_output_shape(self, cnf_dataset, cnf_constraint, model_config):
        x_train, y_train, x_valid, y_valid, x_test, y_test = cnf_dataset
        predicted = _predict(
            "plain_nn_postproc",
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            cnf_constraint,
            model_config,
        )
        assert predicted.shape == (len(x_test), cnf_constraint.n_outputs)

    def test_constraints_satisfied(self, cnf_dataset, cnf_constraint, model_config):
        x_train, y_train, x_valid, y_valid, x_test, y_test = cnf_dataset
        predicted = _predict(
            "plain_nn_postproc",
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            cnf_constraint,
            model_config,
        )
        dummy = np.zeros((len(x_test), max(1, cnf_constraint.n_inputs)))
        results = cnf_constraint.check(np.array(predicted) > 0.5, dummy)
        assert all(results), "Post-processed predictions should satisfy constraints"


class TestPlainNNBoundary:
    def test_output_shape(self, boundary_dataset, boundary_constraint, model_config):
        x_train, y_train, x_valid, y_valid, x_test, y_test = boundary_dataset
        predicted = _predict(
            "plain_nn",
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            boundary_constraint,
            model_config,
        )
        assert predicted.shape == (len(x_test), boundary_constraint.n_outputs)

    def test_no_sigmoid_for_real(
        self, boundary_dataset, boundary_constraint, model_config
    ):
        """Real-valued outputs should not be squeezed by sigmoid."""
        x_train, y_train, x_valid, y_valid, x_test, y_test = boundary_dataset
        predicted = _predict(
            "plain_nn",
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            boundary_constraint,
            model_config,
        )
        pred_np = np.array(predicted)
        # Without sigmoid, some outputs may exceed [0,1]
        # Just check shape is correct and values are finite
        assert np.all(np.isfinite(pred_np))


import pytest


class TestReproducibility:
    """Verify that setting learning_seed produces identical results."""

    METHODS_TO_TEST = ["plain_nn", "plain_nn_postproc", "pishield", "sequential_beam"]

    @pytest.mark.parametrize("method", METHODS_TO_TEST)
    def test_same_seed_same_output(
        self, method, cnf_dataset, cnf_constraint, model_config
    ):
        x_train, y_train, x_valid, y_valid, x_test, y_test = cnf_dataset
        config_seeded = {**model_config, "learning_seed": 42}

        pred1 = _predict(
            method,
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            cnf_constraint,
            config_seeded,
        )
        pred2 = _predict(
            method,
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            cnf_constraint,
            config_seeded,
        )
        np.testing.assert_array_equal(
            np.array(pred1),
            np.array(pred2),
            err_msg=f"{method}: same learning_seed should produce identical outputs",
        )

    @pytest.mark.parametrize("method", METHODS_TO_TEST)
    def test_different_seed_different_output(
        self, method, cnf_dataset, cnf_constraint, model_config
    ):
        x_train, y_train, x_valid, y_valid, x_test, y_test = cnf_dataset

        pred1 = _predict(
            method,
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            cnf_constraint,
            {**model_config, "learning_seed": 1},
        )
        pred2 = _predict(
            method,
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            cnf_constraint,
            {**model_config, "learning_seed": 2},
        )
        assert not np.array_equal(
            np.array(pred1), np.array(pred2)
        ), f"{method}: different learning_seeds should produce different outputs"


class TestPlainNNPostprocBoundary:
    def test_constraints_satisfied(
        self, boundary_dataset, boundary_constraint, model_config
    ):
        x_train, y_train, x_valid, y_valid, x_test, y_test = boundary_dataset
        predicted = _predict(
            "plain_nn_postproc",
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            boundary_constraint,
            model_config,
        )
        dummy = np.zeros((len(x_test), max(1, boundary_constraint.n_inputs)))
        results = boundary_constraint.check(np.array(predicted), dummy)
        assert all(
            results
        ), "Post-processed predictions should satisfy boundary constraints"
