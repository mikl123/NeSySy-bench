"""CNF constraint — distant supervision + DeepProbLog semantic loss."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from nesy_approaches.methods import ALL_METHODS, evaluate
from nesy_data_gen.constraints import CNFConstraint
from nesy_data_gen.generators import DatasetGeneratorFiltering
from nesy_data_gen.utils.deeplog_interface import constraint_to_deeplog_module

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = "cpu"  # "cpu" or "cuda"

TEST_SIZE = 0.2
VALID_SIZE = 0.1

CNF_PARAMS = {
    "n_outputs": 15,
    "n_inputs": 10,
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
    "n_samples": 100000,
    "exploration_set_size": 5000,
    "epochs": 500,
    "lr": 0.01,
    "batch_size": 128,
    "distant_supervision": True,
}

TRAIN_PARAMS = {
    "batch_size": 64,
    "epochs": 100,
    "lr": 0.001,
    "hidden_dim": 64,
    "num_layers": 2,
    "patience": 20,
    "device": DEVICE,
    "use_sigmoid": True,  # sigmoid kept for binary outputs + BCELoss
    "is_real": False,  # propositional mode → BCELoss
}

SEED = 42

cnf = CNFConstraint(**CNF_PARAMS)
cnf.generate()
print("Generated CNF Constraint:")
print(cnf.to_str())

generator = DatasetGeneratorFiltering(**DS_PARAMS)
result = generator.generate_dataset(cnf, seed=SEED, device=DEVICE)

X, Y = result["X"], result["Y"]
ds_info = result["dataset_info"]["distant_supervision"]
print(
    f"Unique positive: {ds_info['n_unique_positive']}, Unique negative: {ds_info['n_unique_negative']}"
)

#################################################
# Training

test_size = int(TEST_SIZE * len(X))
valid_size = int(VALID_SIZE * len(X))
train_size = len(X) - test_size - valid_size

X = torch.asarray(X, dtype=torch.float32)
Y = torch.asarray(Y, dtype=torch.float32)
Y_neural = Y[:, :-1]
Y_query = Y[:, -1:]

x_train = X[:train_size]
x_valid = X[train_size : train_size + valid_size]
x_test = X[train_size + valid_size :]

y_neural_train = Y_neural[:train_size]
y_neural_valid = Y_neural[train_size : train_size + valid_size]
y_neural_test = Y_neural[train_size + valid_size :]

y_query_train = Y_query[:train_size]
y_query_valid = Y_query[train_size : train_size + valid_size]
y_query_test = Y_query[train_size + valid_size :]

# Train base NN to predict y_query
learning_method = ALL_METHODS["plain_nn"]

predicted, _ = learning_method(
    x_train,
    y_query_train,
    x_valid,
    y_query_valid,
    x_test,
    cnf,
    TRAIN_PARAMS,
)
metrics = evaluate(predicted, y_query_test, x_test, cnf)

print("\n" + "=" * 50)
print("Evaluation Metrics (plain_nn):")
print("=" * 50)
for key, value in metrics.items():
    print(f"  {key}: {value}")


# # DeepProbLog

# create a constraint module
constraint_module = constraint_to_deeplog_module(
    cnf._constraints[0], CNF_PARAMS["n_outputs"]
)
constraint_module.to(DEVICE)


# train model to predict the first n columns of the output with distant supervision
learning_method = ALL_METHODS["plain_nn_sl"]

predicted = learning_method(
    x_train,
    y_neural_train,
    y_query_train,
    x_valid,
    y_neural_valid,
    y_query_valid,
    x_test,
    constraint_module,
    TRAIN_PARAMS,
)

metrics = evaluate(predicted, y_query_test, x_test, cnf)

print("\n" + "=" * 50)
print("Evaluation Metrics (plain_nn with semantic loss):")
print("=" * 50)
for key, value in metrics.items():
    print(f"  {key}: {value}")
