# Learning Methods

NeSySy provides several learning methods for structured prediction under constraints. All methods are available through the `ALL_METHODS` dictionary in `nesy_approaches/methods.py` and share the same calling convention.

## Calling convention

Every method is called as a function:

```python
from nesy_approaches.methods import ALL_METHODS, evaluate

predicted = ALL_METHODS["method_name"](
    x_train, y_train,
    x_valid, y_valid,
    x_test,
    constraint,   # generated Constraint object
    config,       # dict with training hyperparameters
)

metrics = evaluate(predicted, y_test, x_test, constraint)
```

The method dispatches on `constraint.is_real` (`False` for `CNFConstraint`, `True` for `SMTBoundedIntervalConstraint` and `SMTLinearInequalityConstraint`) to pick the right solver and loss.

**Common config keys:**

| Key | Type | Description |
|-----|------|-------------|
| `hidden_dim` | `int` | Width of hidden layers |
| `num_layers` | `int` | Number of hidden layers |
| `epochs` | `int` | Maximum training epochs |
| `lr` | `float` | Learning rate |
| `batch_size` | `int` | Mini-batch size |
| `patience` | `int` | Early stopping patience |
| `device` | `str` | `"cpu"` or `"cuda"` |

---

## plain_nn

Standard feedforward network with no constraint awareness. Trains on the dataset with BCE loss (CNF) or L1 loss (real-valued). Makes no guarantee that outputs satisfy the constraint.

**Supports:** CNF, real-valued
**Constraint guarantee:** No

```python
config = {"hidden_dim": 64, "num_layers": 2, "epochs": 1000, "lr": 0.001,
          "batch_size": 64, "patience": 20, "device": "cpu"}

predicted, trainer = ALL_METHODS["plain_nn"](
    x_train, y_train, x_valid, y_valid, x_test, cnf, config
)
metrics = evaluate(predicted, y_test, x_test, cnf)
# {"accuracy": ..., "constraint_violation_ratio": ..., "hamming_distance": ..., ...}
```

---

## plain_nn_postproc

Trains the same feedforward network as `plain_nn`, then applies `constraint.fix()` to every test prediction at inference time. The fix projects violating outputs to the nearest feasible point, so all returned predictions satisfy the constraint.

**Supports:** CNF, real-valued
**Constraint guarantee:** Yes (post-hoc, via `fix()`)

```python
predicted = ALL_METHODS["plain_nn_postproc"](
    x_train, y_train, x_valid, y_valid, x_test, cnf, config
)
# all(cnf.check(predicted, x_test)) is True
```

You can also pass a pre-trained `base_trainer` to skip retraining:

```python
predicted_raw, trainer = ALL_METHODS["plain_nn"](...)
predicted_fixed = ALL_METHODS["plain_nn_postproc"](
    x_train, y_train, x_valid, y_valid, x_test, cnf, config,
    base_trainer=trainer,
)
```

---

## pishield

Wraps the feedforward network with a differentiable enforcement layer from [PiShield](https://github.com/mihaela-stoian/PiShield) that corrects outputs during both training and inference. For CNF constraints, the layer is **CCN+** (Coherent-by-Construction Networks); for real-valued constraints, it is **DRL** (Disjunctive Refinement Layer). Both guarantee that all outputs satisfy the constraint by construction.

**Supports:** CNF, real-valued
**Constraint guarantee:** Yes (by construction, during forward pass)

```python
# CNF — uses CCN+ layer
predicted = ALL_METHODS["pishield"](
    x_train, y_train, x_valid, y_valid, x_test, cnf, config
)

# Real-valued — uses DRL layer
predicted = ALL_METHODS["pishield"](
    x_train, y_train, x_valid, y_valid, x_test, smt, config
)
```

---

## sequential_greedy

Two-stage method for CNF constraints. First trains a base feedforward network on the full output vector. Then trains a sequential (autoregressive) model that predicts one output variable at a time, conditioned on all previously predicted variables. At inference, variables are decoded one by one in greedy order. Does not guarantee constraint satisfaction.

**Supports:** CNF only
**Constraint guarantee:** No

```python
predicted = ALL_METHODS["sequential_greedy"](
    x_train, y_train, x_valid, y_valid, x_test, cnf, config
)
```

---

## sequential_beam

Same two-stage architecture as `sequential_greedy` but uses beam search at inference to explore multiple decoding paths and return the highest-scoring complete assignment. Does not guarantee constraint satisfaction but typically achieves higher accuracy than greedy.

**Supports:** CNF only
**Constraint guarantee:** No

```python
predicted = ALL_METHODS["sequential_beam"](
    x_train, y_train, x_valid, y_valid, x_test, cnf, config
)
```

---

## sequential_fix

Same two-stage architecture as `sequential_greedy` but enforces partial constraint satisfaction at each greedy decoding step. After predicting each variable, a SAT solver checks whether the partial assignment is still consistent with the constraint. If not, the predicted value is flipped before moving to the next variable. This guarantees that the final output satisfies the constraint.

**Supports:** CNF only
**Constraint guarantee:** Yes (SAT check at each decoding step)

```python
predicted = ALL_METHODS["sequential_fix"](
    x_train, y_train, x_valid, y_valid, x_test, cnf, config
)
# all(cnf.check(predicted, x_test)) is True
```

---

## Summary

| Method | CNF | Real-valued | Constraint guarantee |
|--------|:---:|:-----------:|:--------------------:|
| `plain_nn` | ✓ | ✓ | No |
| `plain_nn_postproc` | ✓ | ✓ | Yes (post-hoc) |
| `pishield` | ✓ | ✓ | Yes (by construction) |
| `sequential_greedy` | ✓ | — | No |
| `sequential_beam` | ✓ | — | No |
| `sequential_fix` | ✓ | — | Yes (SAT at each step) |
