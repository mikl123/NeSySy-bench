"""SMT bounded interval — plain NN baseline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.example_setting.boundary_config import (
    BOUNDARY_DS_PARAMS,
    BOUNDARY_PARAMS,
    BOUNDARY_TRAIN_PARAMS,
)
from nesy_approaches.methods import ALL_METHODS, evaluate
from nesy_approaches.utils import generate_dataset
from nesy_data_gen.constraints import SMTBoundedIntervalConstraint

# STEP 1: Create and Generate Constraint

smt_bi = SMTBoundedIntervalConstraint(**BOUNDARY_PARAMS)
smt_bi.generate()
print("Generated SMT Bounded-Interval Constraints:")
print(smt_bi.to_str())

# STEP 2: Generate Dataset

x_train, y_train, x_valid, y_valid, x_test, y_test, _, dataset_info = generate_dataset(
    BOUNDARY_DS_PARAMS, smt_bi
)
print(
    f"\nDataset generated: train={len(x_train)}, valid={len(x_valid)}, test={len(x_test)}"
)

# STEP 3: Train and Evaluate with plain_nn

config = BOUNDARY_TRAIN_PARAMS["model"]
predicted, _ = ALL_METHODS["plain_nn"](
    x_train,
    y_train,
    x_valid,
    y_valid,
    x_test,
    smt_bi,
    config,
)
metrics = evaluate(predicted, y_test, x_test, smt_bi)

print("\n" + "=" * 50)
print("Evaluation Metrics (plain_nn):")
print("=" * 50)
for key, value in metrics.items():
    print(f"  {key}: {value}")
