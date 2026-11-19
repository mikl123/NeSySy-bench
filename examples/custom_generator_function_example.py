"""Custom generator — plug your own ``y = f(x)`` callable into ``generate_dataset``."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesy_data_gen.constraints import CNFConstraint
from nesy_data_gen.generators import DatasetGeneratorFixGuided


def make_random_linear_mapping(n_inputs: int, n_outputs: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(n_inputs, n_outputs, generator=g)
    b = torch.randn(n_outputs, generator=g)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x @ W.to(x.device) + b.to(x.device))

    return fn


if __name__ == "__main__":
    cnf = CNFConstraint(n_outputs=10, n_inputs=4, k=3, r=2.0, seed=42)
    cnf.generate()

    gen = DatasetGeneratorFixGuided(
        n_inputs=4, n_outputs=10, n_samples=200, exploration_set_size=2000
    )

    my_fn = make_random_linear_mapping(n_inputs=4, n_outputs=10, seed=0)
    dataset = gen.generate_dataset(cnf, seed=0, generator_function=my_fn)

    X, Y = dataset["X"], dataset["Y"]
    print(f"generated: X={X.shape}, Y={Y.shape}")
    print(f"all satisfy constraint: {all(cnf.check(Y, X))}")
