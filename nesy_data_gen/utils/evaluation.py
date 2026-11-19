import numpy as np
from sklearn.metrics import log_loss, mean_absolute_error

from nesy_data_gen.constants import FLOAT_PRECISION


def evaluate_prep(probs, y_test, constraint=None, x_test=None, test_region=None):
    if test_region is not None:
        results = {}
        for i in set(test_region):
            mask = test_region == i
            x_masked = x_test[mask] if x_test is not None else None
            results[f"region_{int(i)}"] = evaluate_prep(
                probs[mask], y_test[mask], constraint, x_masked
            )
        return results
    else:
        pred_binary = (probs >= 0.5).astype(int)
        metrics = {
            "binary_cross_entropy": round(
                float(log_loss(y_test.reshape((-1,)), probs.reshape((-1,)))), 4
            ),
            "hamming_distance": round(
                float((pred_binary != y_test).astype(int).mean()), 4
            ),
            "accuracy": round(
                float((pred_binary == y_test).all(axis=1).astype(int).mean()), 4
            ),
        }
        if constraint is not None and x_test is not None:
            metrics["constraint_violation_ratio"] = round(
                float(1 - np.mean(constraint.check(pred_binary, x_test))), 4
            )
        return metrics


def evaluate_real(prediction, y_test, constraint=None, x_test=None, test_region=None):
    if test_region is not None:
        results = {}
        for i in set(test_region):
            mask = test_region == i
            x_masked = x_test[mask] if x_test is not None else None
            results[f"region_{int(i)}"] = evaluate_real(
                prediction[mask], y_test[mask], constraint, x_masked
            )
        return results
    else:
        metrics = {
            "mean_absolute_error": round(
                float(np.mean(np.sum(np.abs(prediction - y_test), axis=1))), 4
            )
        }
        if constraint is not None and x_test is not None:
            fixed = constraint.fix(prediction, x_test)
            metrics["constraint_violation_ratio"] = round(
                float(1 - np.mean(constraint.check(prediction, x_test))), 4
            )
            metrics["mean_consistent_distance"] = round(
                float(np.mean(np.sum(np.abs(prediction - fixed), axis=1))), 4
            )
        return metrics


def evaluate_prep_responses(predicted, y_test, x_test, constraint):
    x_test = x_test.numpy(force=True)
    y_test = y_test.numpy(force=True)
    predicted = predicted.numpy(force=True)

    # assert all(constraint.check(y_test, x_test))

    all_metrics_to_save = {}

    all_metrics_to_save["results_per_region"] = evaluate_prep(
        predicted,
        y_test,
        constraint,
        x_test,
        test_region=constraint.get_regions_for_points(x_test),
    )
    all_metrics_to_save.update(evaluate_prep(predicted, y_test, constraint, x_test))

    return all_metrics_to_save


def evaluate_real_responses(predicted, y_test, x_test, constraint):
    x_test = x_test.numpy(force=True)
    y_test = y_test.numpy(force=True)
    predicted = predicted.numpy(force=True)

    all_metrics_to_save = {}

    # assert all(constraint.check(y_test, x_test)), "Some test samples do not satisfy the constraint"

    all_metrics_to_save["results_per_region"] = evaluate_real(
        predicted,
        y_test,
        constraint,
        x_test,
        test_region=constraint.get_regions_for_points(x_test),
    )
    all_metrics_to_save.update(evaluate_real(predicted, y_test, constraint, x_test))

    return all_metrics_to_save
