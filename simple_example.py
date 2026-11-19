import torch
from sklearn.model_selection import train_test_split

from nesy_approaches.methods import ALL_METHODS, evaluate
from nesy_data_gen.constraints import CNFConstraint
from nesy_data_gen.generators import DatasetGeneratorFixGuided

# 1. Generate constraint
cnf = CNFConstraint(n_outputs=10, n_inputs=4, k=3, r=2.0, seed=42)
cnf.generate()

# 2. Generate dataset
gen = DatasetGeneratorFixGuided(n_inputs=4, n_outputs=10, n_samples=1000, epochs=200)
dataset = gen.generate_dataset(cnf, seed=42)
X = torch.as_tensor(dataset["X"], dtype=torch.float32)
Y = torch.as_tensor(dataset["Y"], dtype=torch.float32)

# 3. Split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=0
)

# 4. Train and evaluate
config = {
    "hidden_dim": 64,
    "num_layers": 2,
    "epochs": 500,
    "lr": 0.001,
    "batch_size": 64,
    "patience": 20,
    "device": "cpu",
}

predicted, _ = ALL_METHODS["plain_nn"](
    x_train, y_train, x_val, y_val, x_test, cnf, config
)
print(evaluate(predicted, y_test, x_test, cnf))
