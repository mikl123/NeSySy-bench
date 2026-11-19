"""Training methods registry."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from nesy_approaches.model1 import Model1Trainer
from nesy_approaches.sequential_model import SequentialTrainer, build_sequential_dataset
from nesy_data_gen.utils.evaluation import (
    evaluate_prep_responses,
    evaluate_real_responses,
)


def _build_shield(constraint):
    """Build the appropriate PiShield layer for the constraint type."""
    from pishield.shield_layer import build_shield_layer

    requirements_text = constraint._to_pishield_str_one(0)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write(requirements_text)
        path = tmp.name
    try:
        if not constraint.is_real:
            CL = build_shield_layer(
                constraint.n_outputs,
                path,
                ordering_choice="degree",
                requirements_type="propositional",
            )
        else:
            CL = build_shield_layer(
                constraint.n_outputs, path, requirements_type="qflra"
            )
    finally:
        os.unlink(path)
    return CL


def evaluate(predicted, y_test, x_test, constraint) -> dict:
    """Run the appropriate evaluation and return flat scalar metrics."""
    if not isinstance(predicted, torch.Tensor):
        predicted = torch.tensor(np.array(predicted), dtype=torch.float32)
    if not constraint.is_real:
        metrics = evaluate_prep_responses(predicted, y_test, x_test, constraint)
    else:
        metrics = evaluate_real_responses(predicted, y_test, x_test, constraint)
    return metrics


# Available learning methods


def plain_nn(x_train, y_train, x_valid, y_valid, x_test, constraint, config):
    """Standard feedforward NN, no constraint enforcement. Returns (predicted, trainer)."""
    trainer = Model1Trainer({**config, "is_real": constraint.is_real})
    trainer.train(x_train, y_train, x_valid, y_valid)
    return trainer.inference(x_test).detach(), trainer


def plain_nn_postproc(
    x_train,
    y_train,
    x_valid,
    y_valid,
    x_test,
    constraint,
    config,
    base_trainer=None,
):
    """Plain NN + post-hoc constraint.fix() correction."""
    if base_trainer is not None:
        predicted = base_trainer.inference(x_test).detach()
    else:
        predicted, _ = plain_nn(
            x_train, y_train, x_valid, y_valid, x_test, constraint, config
        )
    fixed = constraint.fix(np.array(predicted), np.array(x_test))
    return np.array(fixed)


def plain_nn_sl(
    x_train,
    y_neural_train,
    y_query_train,
    x_valid,
    y_neural_valid,
    y_query_valid,
    x_test,
    sl_module,
    config,
):
    """Standard feedforward NN, with semantic loss. CNF-only (used for DeepProbLog-style distant supervision)."""
    trainer = Model1Trainer({**config, "is_real": False})
    trainer.train_with_sl(
        x_train,
        y_neural_train,
        y_query_train,
        x_valid,
        y_neural_valid,
        y_query_valid,
        sl_module,
    )
    neural_prediction = trainer.inference(x_test)
    query_prediction = sl_module(neural_prediction)
    query_prediction = torch.clamp(query_prediction, min=0.0, max=1.0)
    return query_prediction.detach()


def pishield(x_train, y_train, x_valid, y_valid, x_test, constraint, config):
    """NN wrapped with PiShield correction layer (DRL for QFLRA, CCN+ for propositional)."""
    CL = _build_shield(constraint)
    trainer = Model1Trainer(
        {**config, "is_real": constraint.is_real, "shield_layer": CL}
    )
    trainer.train(x_train, y_train, x_valid, y_valid)
    return trainer.inference(x_test).detach()


def _train_base_and_sequential(
    x_train,
    y_train,
    x_valid,
    y_valid,
    x_test,
    constraint,
    config,
    base_trainer=None,
):
    """Shared setup: train base NN once, then train sequential model.

    Returns (seq_trainer, base_preds_test, n_classes).
    """
    # 1. Train base NN (or reuse provided one)
    if base_trainer is None:
        base_trainer = Model1Trainer({**config, "is_real": constraint.is_real})
        base_trainer.train(x_train, y_train, x_valid, y_valid)

    # 2. Inference on all splits
    base_preds_train = np.array(
        base_trainer.inference(x_train).detach(), dtype=np.float32
    )
    base_preds_valid = np.array(
        base_trainer.inference(x_valid).detach(), dtype=np.float32
    )
    base_preds_test = np.array(
        base_trainer.inference(x_test).detach(), dtype=np.float32
    )

    n_classes = y_train.shape[1]
    y_train_np = np.array(y_train, dtype=np.float32)
    y_valid_np = np.array(y_valid, dtype=np.float32)

    # 3. Build sequential datasets
    feat_train, tgt_train = build_sequential_dataset(base_preds_train, y_train_np)
    feat_valid, tgt_valid = build_sequential_dataset(base_preds_valid, y_valid_np)

    # 4. Train sequential model (model built inside trainer, after seeding)
    trainer = SequentialTrainer(model=None, config=config)
    trainer.fit(feat_train, tgt_train, feat_valid, tgt_valid)

    return trainer, base_preds_test, n_classes


def sequential_greedy(
    x_train,
    y_train,
    x_valid,
    y_valid,
    x_test,
    constraint,
    config,
    base_trainer=None,
):
    """Base NN → sequential autoregressive model (greedy decode)."""
    trainer, base_preds_test, n_classes = _train_base_and_sequential(
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        constraint,
        config,
        base_trainer=base_trainer,
    )
    return trainer.greedy_predict(base_preds_test, n_classes)


def sequential_beam(
    x_train,
    y_train,
    x_valid,
    y_valid,
    x_test,
    constraint,
    config,
    base_trainer=None,
):
    """Base NN → sequential autoregressive model (beam decode)."""
    trainer, base_preds_test, n_classes = _train_base_and_sequential(
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        constraint,
        config,
        base_trainer=base_trainer,
    )
    return [i[0][0] for i in trainer.beam_predict(base_preds_test, n_classes)]


def sequential_fix(
    x_train,
    y_train,
    x_valid,
    y_valid,
    x_test,
    constraint,
    config,
    base_trainer=None,
):
    """Base NN → sequential model + constraint fixing at each greedy step."""
    trainer, base_preds_test, n_classes = _train_base_and_sequential(
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        constraint,
        config,
        base_trainer=base_trainer,
    )
    return trainer.greedy_predict_with_fix(
        base_preds_test,
        n_classes,
        constraint,
        x_test,
    )


def ecc(x_train, y_train, x_valid, y_valid, x_test, constraint, config):
    """Ensemble of Classifier Chains (multi-label baseline, CNF only)."""
    from scipy.sparse import issparse
    from sklearn.naive_bayes import GaussianNB
    from skmultilearn.problem_transform import ClassifierChain

    x_tr = np.array(x_train, dtype=np.float64)
    y_tr = np.array(y_train, dtype=np.int32)
    x_te = np.array(x_test, dtype=np.float64)

    ensemble_size = config.get("ecc_ensemble_size", 10)
    chains = []
    for i in range(ensemble_size):
        cc = ClassifierChain(
            classifier=GaussianNB(),
            require_dense=[True, True],
            order=np.random.RandomState(i).permutation(y_tr.shape[1]).tolist(),
        )
        cc.fit(x_tr, y_tr)
        chains.append(cc)

    # Majority vote across chains
    preds = []
    for cc in chains:
        p = cc.predict(x_te)
        preds.append(p.toarray() if issparse(p) else np.array(p))
    predicted = (np.mean(preds, axis=0) >= 0.5).astype(np.float32)
    return predicted


def rakel(x_train, y_train, x_valid, y_valid, x_test, constraint, config):
    """RAKEL — Random k-Labelsets (multi-label baseline, CNF only)."""
    from sklearn.naive_bayes import GaussianNB
    from skmultilearn.ensemble import RakelO

    x_tr = np.array(x_train, dtype=np.float64)
    y_tr = np.array(y_train, dtype=np.int32)
    x_te = np.array(x_test, dtype=np.float64)

    n_labels = y_tr.shape[1]
    k = config.get("rakel_k", 3)
    base_clf = GaussianNB()
    model = RakelO(
        base_classifier=base_clf,
        base_classifier_require_dense=[True, True],
        labelset_size=min(k, n_labels),
        model_count=2 * n_labels,
    )
    model.fit(x_tr, y_tr)
    predicted = model.predict(x_te).toarray().astype(np.float32)
    return predicted


def decision_tree(x_train, y_train, x_valid, y_valid, x_test, constraint, config):
    """Decision tree baseline. Regressor for real-valued, classifier for CNF."""
    from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    learning_seed = config.get("learning_seed", 42)

    x_tr = np.array(x_train, dtype=np.float64)
    y_tr = np.array(y_train, dtype=np.float64)
    x_te = np.array(x_test, dtype=np.float64)

    if not constraint.is_real:
        model = MultiOutputClassifier(
            DecisionTreeClassifier(random_state=learning_seed)
        )
        model.fit(x_tr, y_tr.astype(int))
        predicted = model.predict(x_te).astype(np.float32)
    else:
        model = MultiOutputRegressor(DecisionTreeRegressor(random_state=learning_seed))
        model.fit(x_tr, y_tr)
        predicted = np.clip(model.predict(x_te), 0, 1).astype(np.float32)
    return predicted


ALL_METHODS = {
    "plain_nn": plain_nn,
    "plain_nn_postproc": plain_nn_postproc,
    "plain_nn_sl": plain_nn_sl,
    "pishield": pishield,
    "sequential_beam": sequential_beam,
    "sequential_greedy": sequential_greedy,
    "sequential_fix": sequential_fix,
    "ecc": ecc,
    "rakel": rakel,
    "decision_tree": decision_tree,
}
