# Constraints

## Constraint (Base Class)

Base class for all constraints. Handles region partitioning — the input space is split into bands so different parts of the input space can have different constraints.

**Parameters:**
- `c` *(int, default=1)* — number of unique constraint formulas to generate
- `b` *(int, default=1)* — number of input regions per constraint formula (`c × b` total regions)
- `n_outputs` *(int, default=10)* — number of output variables
- `n_inputs` *(int, default=0)* — number of input variables (needed when `use_input_variables=True`)
- `seed` *(int, default=0)* — random seed
- `use_input_variables` *(bool, default=False)* — if True, input variables appear inside the constraint alongside output variables
- `require_all_inputs_valid` *(bool, default=True)* — if True, every possible input value must have at least one valid output (only meaningful when `use_input_variables=True`)

**Methods:**
- `generate()` — create the constraint formulas and assign them to regions; call this before anything else
- `check(vectors, inputs)` — returns a list of booleans, one per sample, True if the output satisfies its constraint
- `fix(vectors, inputs)` — projects each violating output to the nearest satisfying point; valid outputs are returned unchanged
- `to_str()` — returns a human-readable formula for each constraint with its input region bounds
- `save_state(dir_path)` / `Constraint.load_state(dir_path)` — save and reload a generated constraint from disk

```python
import numpy as np
from nesy_data_gen.constraints import CNFConstraint

cnf = CNFConstraint(n_outputs=5, k=3, r=2.0, seed=0)
cnf.generate()

# Some example outputs (binary vectors)
outputs = np.array([[1, 0, 1, 0, 1], [0, 0, 0, 0, 0]], dtype=np.float32)
inputs  = np.array([[0.2, 0.5, 0.1, 0.3], [0.8, 0.4, 0.9, 0.2]], dtype=np.float32)

print(cnf.check(outputs, inputs))   # [True, False]

fixed = cnf.fix(outputs, inputs)
print(cnf.check(fixed, inputs))     # [True, True]
```

---

## CNFConstraint (Constraint)

Generates k-SAT constraints over Boolean output variables. Each formula is a conjunction of clauses; each clause is a disjunction of k literals (`x_i` or `¬x_i`). Generation uses a planted satisfying assignment so every formula is guaranteed satisfiable.

**Parameters:**
- `k` *(int, default=3)* — number of literals per clause (3 gives standard 3-SAT)
- `r` *(float, default=3.0)* — clause-to-variable ratio; controls constraint tightness (`n_clauses = n_vars × r`). Near `r ≈ 4.267` the problem is hardest.

```python
from nesy_data_gen.constraints import CNFConstraint

cnf = CNFConstraint(n_outputs=10, k=3, r=2.0, c=2, b=1, seed=42)
cnf.generate()

print(cnf)
# CNF(3-SAT, 20 clauses, 10 vars [10O], V=O, 2x1 regions)

print(cnf.to_str())
# [R0:[0.000,0.500]]
#  (x1 ∨ ¬x3 ∨ x7)
#   ∧ (¬x2 ∨ x5 ∨ ¬x9)
#   ∧ ...
#
# [R1:[0.500,1.000]]
#  (x2 ∨ x4 ∨ ¬x6)
#   ∧ ...
```

---

## SMTBoundedIntervalConstraint (Constraint)

First generates a CNF formula (using the same `k` and `r` parameters), then uses it as a backbone: every Boolean literal in that CNF is replaced by a bounded interval condition `c_min ≤ x_i ≤ c_max` with random bounds in [0, 1]. The result is a real-valued constraint with the same clause structure as the CNF backbone but over continuous outputs. Checking and fixing use the Z3 solver.

**Parameters:**
- `k` *(int, default=3)* — literals per clause in the CNF backbone
- `r` *(float, default=3.0)* — clause-to-variable ratio for the CNF backbone
- `min_gap` *(float, default=0.1)* — minimum width of each interval `[c_min, c_max]`; larger values make individual intervals easier to satisfy

```python
from nesy_data_gen.constraints import SMTBoundedIntervalConstraint

smt_bi = SMTBoundedIntervalConstraint(n_outputs=10, k=3, r=1.5, min_gap=0.15, c=1, b=1, seed=0)
smt_bi.generate()

print(smt_bi)
# SMTBoundedInterval(10O vars, V=O, 1x1 regions)

print(smt_bi.to_str())
# [R0:[0.000,1.000]]
#  (0.12 ≤ x1 ≤ 0.45 ∨ 0.60 ≤ x3 ≤ 0.81 ∨ 0.30 ≤ x7 ≤ 0.72)
#   ∧ (0.05 ≤ x2 ≤ 0.33 ∨ ...)
#   ∧ ...
```

---

## SMTLinearInequalityConstraint (Constraint)

First generates a CNF formula (using the same `k` and `r` parameters), then uses it as a backbone: every Boolean literal in that CNF is replaced by a linear inequality `c₀ + c₁·x_j1 + c₂·x_j2 + … ≤ 0` with random integer coefficients. Trivially always-true or always-false inequalities are discarded and resampled. Checking and fixing use Z3.

**Parameters:**
- `k` *(int, default=3)* — literals per clause in the CNF backbone
- `r` *(float, default=3.0)* — clause-to-variable ratio for the CNF backbone
- `coef_abs_max` *(int, default=5)* — maximum absolute value of sampled integer coefficients
- `n_terms_min` *(int, default=1)* — minimum number of variables per inequality
- `n_terms_max` *(int, default=5)* — maximum number of variables per inequality

```python
from nesy_data_gen.constraints import SMTLinearInequalityConstraint

smt_li = SMTLinearInequalityConstraint(
    n_outputs=10, k=3, r=2.0,
    coef_abs_max=3, n_terms_min=2, n_terms_max=4,
    c=1, b=1, seed=7,
)
smt_li.generate()

print(smt_li)
# SMTLinearInequality(10O vars, coef_max=3, V=O, 1x1 regions)

print(smt_li.to_str())
# [R0:[0.000,1.000]]
#  (2 + 3·x2 + -1·x5 ≤ 0 ∨ -2 + 1·x1 + 2·x8 ≤ 0 ∨ ...)
#   ∧ ...
```

---

## Writing a Custom Constraint

To add a new constraint, subclass `Constraint` together with a Pydantic config that extends `BaseConstraintConfig`, and override three abstract hooks defined in [`base.py`](../nesy_data_gen/constraints/base.py):

- `_generate_single(constraint_idx)` — produce one constraint instance (called `c` times by `generate()`)
- `_check_one(vector, input, constraint)` — return `True` iff a single output vector satisfies its constraint
- `_to_str_one(constraint)` — human-readable formula for `to_str()`

`_fix_one(vector, input, constraint)` is optional — override it to enable `.fix()` (the default is a no-op).

### Minimal example

A toy real-valued constraint requiring the sum of outputs to be at least a threshold lives in [`examples/custom_constraint_example.py`](../examples/custom_constraint_example.py) and runs as-is:

```bash
python examples/custom_constraint_example.py
```

The relevant subclass:

```python
class ThresholdSumConstraint(Constraint, ThresholdSumConstraintConfig):
    """Sum of output variables must be >= a per-constraint random threshold."""

    def _generate_single(self, constraint_idx: int) -> float:
        rng = np.random.default_rng(self.seed + constraint_idx)
        return float(rng.uniform(self.threshold_min, self.threshold_max))

    def _check_one(self, vector, input, constraint) -> bool:
        return float(np.sum(vector)) >= constraint

    def _to_str_one(self, constraint) -> str:
        return f"sum(x) >= {constraint:.3f}"

    def _fix_one(self, vector, input, constraint):
        v = np.asarray(vector, dtype=np.float32).copy()
        deficit = constraint - float(v.sum())
        if deficit <= 0:
            return v
        return np.clip(v + deficit / len(v), 0.0, 1.0)
```
