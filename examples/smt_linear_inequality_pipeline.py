"""SMT linear inequality — plain NN baseline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.example_setting.linear_inequality_config import (
    LINEAR_INEQUALITY_DS_PARAMS,
    LINEAR_INEQUALITY_PARAMS,
    LINEAR_INEQUALITY_TRAIN_PARAMS,
)
from nesy_approaches.methods import ALL_METHODS, evaluate
from nesy_approaches.utils import generate_dataset
from nesy_data_gen.constraints import SMTLinearInequalityConstraint

# STEP 1: Create and Generate Constraint

smt_li = SMTLinearInequalityConstraint(**LINEAR_INEQUALITY_PARAMS)
smt_li.generate()
print("Generated SMT Linear Inequality Constraints:")
print(smt_li.to_str())

# STEP 2: Generate Dataset

x_train, y_train, x_valid, y_valid, x_test, y_test, _, dataset_info = generate_dataset(
    LINEAR_INEQUALITY_DS_PARAMS, smt_li
)
print(
    f"\nDataset generated: train={len(x_train)}, valid={len(x_valid)}, test={len(x_test)}"
)

# STEP 3: Train and Evaluate with plain_nn

config = LINEAR_INEQUALITY_TRAIN_PARAMS["model"]
predicted, _ = ALL_METHODS["plain_nn"](
    x_train,
    y_train,
    x_valid,
    y_valid,
    x_test,
    smt_li,
    config,
)
metrics = evaluate(predicted, y_test, x_test, smt_li)

print("\n" + "=" * 50)
print("Evaluation Metrics (plain_nn):")
print("=" * 50)
for key, value in metrics.items():
    print(f"  {key}: {value}")
