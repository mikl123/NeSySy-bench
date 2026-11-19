"""Tests for the sequential model components."""

import numpy as np
import torch

from nesy_approaches.sequential_model import (
    SequentialModel,
    build_sequential_dataset,
    generate_valuations,
)


class TestGenerateValuations:
    def test_shape(self):
        actuals = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        partial = generate_valuations(actuals)
        assert partial.shape == (6, 2)

    def test_teacher_forcing(self):
        actuals = np.array([[1, 0, 1]], dtype=np.float32)
        partial = generate_valuations(actuals)
        np.testing.assert_array_equal(partial[0], [-1, -1])
        np.testing.assert_array_equal(partial[1], [1, -1])
        np.testing.assert_array_equal(partial[2], [1, 0])


class TestBuildSequentialDataset:
    def test_shapes(self):
        base_preds = np.array([[0.8, 0.2, 0.6]], dtype=np.float32)
        y = np.array([[1, 0, 1]], dtype=np.float32)
        features, targets = build_sequential_dataset(base_preds, y)
        assert features.shape[0] == 3
        assert targets.shape == (3,)
        assert features.shape[1] == 5

    def test_targets_are_flattened_labels(self):
        y = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        base_preds = np.random.rand(2, 3).astype(np.float32)
        _, targets = build_sequential_dataset(base_preds, y)
        np.testing.assert_array_equal(targets, [1, 0, 1, 0, 1, 0])


class TestSequentialModel:
    def test_forward_shape(self):
        model = SequentialModel(input_dim=5, hidden_size=8, num_layers=3)
        x = torch.randn(4, 5)
        out = model(x)
        assert out.shape == (4,)

    def test_output_range(self):
        model = SequentialModel(input_dim=5, hidden_size=8, num_layers=3)
        x = torch.randn(100, 5)
        out = model(x).detach().numpy()
        assert np.all(out >= 0) and np.all(out <= 1)


class TestSequentialMethods:
    """Test sequential methods (beam, fix) via ALL_METHODS on CNF constraints."""

    def test_sequential_greedy_binary(self, cnf_dataset, cnf_constraint, model_config):
        from nesy_approaches.methods import ALL_METHODS

        x_train, y_train, x_valid, y_valid, x_test, y_test = cnf_dataset
        predicted = ALL_METHODS["sequential_greedy"](
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            cnf_constraint,
            model_config,
        )
        for row in predicted:
            assert set(row).issubset({0, 1, 0.0, 1.0})
        assert len(predicted) == len(x_test)
        assert len(predicted[0]) == cnf_constraint.n_outputs

    def test_sequential_beam_binary(self, cnf_dataset, cnf_constraint, model_config):
        from nesy_approaches.methods import ALL_METHODS

        x_train, y_train, x_valid, y_valid, x_test, y_test = cnf_dataset
        predicted = ALL_METHODS["sequential_beam"](
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            cnf_constraint,
            model_config,
        )
        for row in predicted:
            assert set(row).issubset({0, 1, 0.0, 1.0})
        assert len(predicted) == len(x_test)
        assert len(predicted[0]) == cnf_constraint.n_outputs

    def test_sequential_fix_all_valid(self, cnf_dataset, cnf_constraint, model_config):
        from nesy_approaches.methods import ALL_METHODS

        x_train, y_train, x_valid, y_valid, x_test, y_test = cnf_dataset
        predicted = ALL_METHODS["sequential_fix"](
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            cnf_constraint,
            model_config,
        )

        assert predicted.shape == (len(x_test), cnf_constraint.n_outputs)
        dummy = np.zeros((len(x_test), max(1, cnf_constraint.n_inputs)))
        checks = cnf_constraint.check(np.array(predicted), dummy)
        assert all(
            checks
        ), f"sequential_fix should produce all valid outputs, got {sum(checks)}/{len(checks)}"
