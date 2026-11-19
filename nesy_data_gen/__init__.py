"""
Neuro-Symbolic Dataset Generator

A framework for generating datasets that combine neural and symbolic AI approaches.
"""

__version__ = "0.1.0"

from nesy_data_gen.constraints import (
    CNFConstraint,
    SMTBoundedIntervalConstraint,
    SMTConstraint,
    SMTLinearInequalityConstraint,
)
from nesy_data_gen.generators import DatasetGenerator, DatasetGeneratorFiltering

__all__ = [
    "__version__",
    # Generators
    "DatasetGenerator",
    "DatasetGeneratorFiltering",
    # Constraints
    "CNFConstraint",
    "SMTConstraint",
    "SMTBoundedIntervalConstraint",
    "SMTLinearInequalityConstraint",
]
