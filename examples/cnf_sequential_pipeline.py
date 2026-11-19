"""CNF constraint — sequential autoregressive model."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.example_setting.cnf_config import (
    CNF_DS_PARAMS,
    CNF_PARAMS,
    CNF_TRAIN_PARAMS,
)
from nesy_approaches.methods import ALL_METHODS, evaluate
from nesy_approaches.utils import generate_dataset
from nesy_data_gen.constraints import CNFConstraint

# STEP 1: Create and Generate Constraint

cnf = CNFConstraint(**CNF_PARAMS)
cnf.generate()
print("Generated CNF Constraints:")
print(cnf.to_str())

# STEP 2: Generate Dataset

x_train, y_train, x_valid, y_valid, x_test, y_test, _, dataset_info = generate_dataset(
    CNF_DS_PARAMS, cnf
)
print(
    f"\nDataset generated: train={len(x_train)}, valid={len(x_valid)}, test={len(x_test)}"
)

# STEP 3: Train and Evaluate with sequential_fix

config = CNF_TRAIN_PARAMS["model"]
predicted = ALL_METHODS["sequential_beam"](
    x_train,
    y_train,
    x_valid,
    y_valid,
    x_test,
    cnf,
    config,
)
metrics = evaluate(predicted, y_test, x_test, cnf)

print("\n" + "=" * 50)
print("Evaluation Metrics (sequential):")
print("=" * 50)
for key, value in metrics.items():
    print(f"  {key}: {value}")
