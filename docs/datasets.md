# Dataset Generation

## DatasetGenerator (Base Class)

Base class for all dataset generators. First trains a small neural network (the *base mapping*) on random inputs to produce a diverse spread of candidate outputs, then collects constraint-satisfying samples from each input region using the subclass's alignment strategy.

**Parameters:**
- `n_inputs` *(int)* — number of input features (must match the constraint)
- `n_outputs` *(int)* — number of output variables (must match the constraint)
- `n_samples` *(int, default=500)* — total samples to generate, split equally across regions
- `hidden_dim` *(int, default=64)* — width of each hidden layer in the base mapping network
- `num_hidden` *(int, default=2)* — number of hidden layers in the base mapping network
- `epochs` *(int, default=1000)* — maximum training epochs for the base mapping
- `lr` *(float, default=0.001)* — learning rate
- `batch_size` *(int, default=512)* — mini-batch size during base mapping training
- `exploration_set_size` *(int, default=10000)* — number of random inputs used to train the base mapping
- `losses` *(list[LossConfig], default=[soft_repulsion])* — pretraining loss components controlling the base mapping training objective
- `distant_supervision` *(bool, default=False)* — if True, also collect non-satisfying outputs and append a label (True for satisfiable and False) column to Y (CNF only)

**Methods:**
- `generate_dataset(constraint, seed)` — run the full pipeline and return a dict with keys `X`, `Y`, `constraint`, and `dataset_info`

**`dataset_info` keys:**

| Key | Description |
|-----|-------------|
| `n_samples_generated` | Actual number of samples collected (may be less than `n_samples` for Filtering) |
| `deficit_samples` | True if fewer samples than requested were returned |
| `first_valid_set_size` | List of `(region_idx, count)` — how many valid outputs the base mapping produced per region before resampling |
| `n_samples_per_region` | List of `(region_idx, count)` — final sample count per region |
| `mcd` | Mean pairwise cosine distance across all output samples — higher is more diverse |
| `validity_rate` | Fraction of outputs that satisfy the constraint (always 1.0 after generation) |
| `per_region_metrics` | Same metrics broken down per region |
| `base_model_training` | JSON string with base mapping training stats (epochs, best loss, per-loss history) |
| `fix_distance_mean` | *(Fix-Guided only)* Mean L1 distance between raw and fixed outputs per region |
| `n_noisy_valid` | *(Fix-Guided only)* Samples rescued from the boundary by Gaussian perturbation per region |
| `n_on_boundary` | *(Fix-Guided only)* Samples that stayed on the constraint boundary as fallback per region |

**Losses** control what the base mapping is trained to do:

| Name | Effect |
|------|--------|
| `soft_repulsion` | Pushes output samples apart — the primary diversity signal (default) |
| `decorrelation` | Penalises linear dependence between output dimensions |
| `small_entropy` | Pushes each output toward 0 or 1 — useful for Boolean/CNF outputs |
| `large_entropy` | Pushes the mean of each output dimension toward 0.5 — balances classes |
| `l1` | L1 weight regularisation |
| `constraint` | Pulls outputs toward the nearest feasible point using `fix()` |

```python
from nesy_data_gen.generators.config import LossConfig

losses = [
    LossConfig(name="soft_repulsion", weight=1.0),
    LossConfig(name="small_entropy",  weight=0.5),
]
```

### Using a Custom Generator Function

Pass any callable mapping `(N, n_inputs)` to `(N, n_outputs)` via the `generator_function` kwarg of `generate_dataset` to skip the diversity-training stage and use it directly as the trained base mapping. The callable receives a `torch.Tensor` on `device` and must return a tensor (or array — coerced).

```python
import torch
from nesy_data_gen.constraints import CNFConstraint
from nesy_data_gen.generators import DatasetGeneratorFixGuided

W = torch.randn(4, 10)
b = torch.randn(10)
my_fn = lambda x: torch.sigmoid(x @ W.to(x.device) + b.to(x.device))

cnf = CNFConstraint(n_outputs=10, n_inputs=4, k=3, r=2.0, seed=42); cnf.generate()
gen = DatasetGeneratorFixGuided(n_inputs=4, n_outputs=10, n_samples=200)
dataset = gen.generate_dataset(cnf, seed=0, generator_function=my_fn)
```

Full runnable example: [`examples/custom_generator_function_example.py`](../examples/custom_generator_function_example.py).

---

## DatasetGeneratorFiltering (DatasetGenerator)

Keeps only outputs that already satisfy the constraint without modifying them. After the base mapping pass, it resamples near valid outputs using Gaussian noise to fill the quota.

**Parameters:**
- `max_iterations` *(int, default=3)* — how many resampling rounds to run near valid points if the initial pass doesn't fill the quota
- `noise_std` *(float, default=0.2)* — standard deviation of Gaussian noise when resampling near valid points

May return fewer than `n_samples` if the feasible region is very small. Use when output quality matters more than a guaranteed count.

```python
from nesy_data_gen.constraints import CNFConstraint
from nesy_data_gen.generators import DatasetGeneratorFiltering

cnf = CNFConstraint(n_outputs=10, k=3, r=2.0, seed=0)
cnf.generate()

gen = DatasetGeneratorFiltering(
    n_inputs=4, n_outputs=10, n_samples=1000,
    max_iterations=5, noise_std=0.15,
)
dataset = gen.generate_dataset(cnf, seed=42)

X, Y = dataset["X"], dataset["Y"]
print(X.shape, Y.shape)   # (1000, 4)  (1000, 10)
print(all(cnf.check(Y, X)))  # True
```

---

## DatasetGeneratorFixGuided (DatasetGenerator)

Projects every output to the nearest feasible point using `constraint.fix()`, then tries to move boundary-fixed points into the interior by adding small Gaussian perturbations. Always returns exactly `n_samples`.

**Parameters:**
- `fix_noise_std` *(float, default=0.3)* — standard deviation of Gaussian noise when perturbing boundary-fixed outputs inward
- `fix_iterations` *(int, default=20)* — how many perturbation rounds to attempt per region

Use when an exact sample count is required or when the feasible region is too sparse for filtering to fill the quota. Trade-off: some outputs may land on the constraint boundary if perturbation fails.

```python
from nesy_data_gen.constraints import SMTBoundedIntervalConstraint
from nesy_data_gen.generators import DatasetGeneratorFixGuided

smt = SMTBoundedIntervalConstraint(n_outputs=10, k=3, r=1.5, min_gap=0.15, seed=0)
smt.generate()

gen = DatasetGeneratorFixGuided(
    n_inputs=4, n_outputs=10, n_samples=1000,
    fix_noise_std=0.2, fix_iterations=30,
)
dataset = gen.generate_dataset(smt, seed=42)

X, Y = dataset["X"], dataset["Y"]
print(X.shape, Y.shape)      # (1000, 4)  (1000, 10)
print(all(smt.check(Y, X)))  # True
```
