"""CNF constraint — distant-supervision dataset generation (no training)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesy_data_gen.constraints import CNFConstraint
from nesy_data_gen.generators import DatasetGeneratorFiltering

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CNF_PARAMS = {
    "n_outputs": 10,
    "n_inputs": 5,
    "c": 1,
    "b": 1,
    "seed": 41,
    "k": 3,
    "r": 1,
    "use_input_variables": False,
}

DS_PARAMS = {
    "n_inputs": CNF_PARAMS["n_inputs"],
    "n_outputs": CNF_PARAMS["n_outputs"],
    "n_samples": 10000,
    "exploration_set_size": 5000,
    "epochs": 500,
    "lr": 0.01,
    "batch_size": 128,
    "distant_supervision": True,
}

SEED = 42

cnf = CNFConstraint(**CNF_PARAMS)
cnf.generate()
print("Generated CNF Constraint:")
print(cnf.to_str())

generator = DatasetGeneratorFiltering(**DS_PARAMS)
result = generator.generate_dataset(cnf, seed=SEED)

X, Y = result["X"], result["Y"]
ds_info = result["dataset_info"]["distant_supervision"]
print(
    f"Unique positive: {ds_info['n_unique_positive']}, Unique negative: {ds_info['n_unique_negative']}"
)
