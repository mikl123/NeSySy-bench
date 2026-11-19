"""Base class for constraints."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


class Constraint(ABC):
    """Abstract base class for all constraint types.

    Supports different constraints for different input regions.
    Constraints stored in memory.
    Subclasses must implement: check, generate_single, save_to_file, fix
    """

    def __init__(self, *args, **kwargs) -> None:
        self._constraints: dict[int, Any] = {}  # constraint_idx -> constraint data
        self._region_to_constraint: dict[int, int] = {}  # region_idx -> constraint_idx
        self._region_bounds: dict[int, tuple[float, float]] = (
            {}
        )  # region_idx -> (low, high)
        self._file_paths: dict[int, str] = {}  # constraint_idx -> file path
        self._sample_regions: np.ndarray = np.array([])
        self._bounds: np.ndarray = None  # array of boundary values
        super().__init__(*args, **kwargs)

    def generate(self, region_mapping: list[int] | None = None) -> None:
        """Generate constraints for all regions.

        Total regions = c * b, but only c unique constraints.
        Mapping is randomized unless region_mapping is provided.
        Also calculates input bounds for each region based on first feature.

        Args:
            region_mapping: Optional explicit list mapping region_idx -> constraint_idx.
                Must have length c * b. If None, mapping is shuffled based on seed.
        """
        self._constraints = {}
        self._region_to_constraint = {}
        self._region_bounds = {}

        # Calculate input bounds for regions (based on first feature)
        n_regions = self.c * self.b
        self._bounds = np.linspace(0, 1, n_regions + 1)

        # Store bounds for each region
        for region_idx in range(n_regions):
            self._region_bounds[region_idx] = (
                self._bounds[region_idx],
                self._bounds[region_idx + 1],
            )

        # Generate c unique constraints
        for constraint_idx in range(self.c):
            constraint_data = self._generate_single(constraint_idx)
            self._constraints[constraint_idx] = constraint_data

        if region_mapping is not None:
            constraint_indices = list(region_mapping)
        else:
            # Create list: each constraint appears b times
            constraint_indices = [i for i in range(self.c) for _ in range(self.b)]
            # Shuffle randomly
            np.random.seed(self.seed)
            np.random.shuffle(constraint_indices)

        # Map regions to constraints
        for region_idx, constraint_idx in enumerate(constraint_indices):
            self._region_to_constraint[region_idx] = constraint_idx

    @abstractmethod
    def _generate_single(self, constraint_idx: int) -> Any:
        """Generate a single constraint.

        Args:
            constraint_idx: Index of the constraint.

        Returns:
            Constraint data (stored in memory).
        """
        pass

    def save_to_file(self, exp_path: str) -> dict[int, str]:
        """Save all constraints to files.

        Args:
            exp_path: Path to save files.

        Returns:
            Dictionary mapping region_idx to file path.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement save_to_file. "
            "Use save_state() instead."
        )

    def get_constraints_for_samples(self, regions: np.ndarray) -> list[Any]:
        """Get constraint data for each sample based on their regions."""
        return [self._constraints[self._region_to_constraint[int(r)]] for r in regions]

    def get_constraints_for_points(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, list[Any]]:
        """Get both regions and constraints for points in a single call.

        Args:
            points: Array of points (N, D) where first feature determines region.

        Returns:
            Tuple of (regions array, constraints list).
        """
        regions = self.get_regions_for_points(points)
        constraints = self.get_constraints_for_samples(regions)
        return constraints

    def get_regions_for_points(self, points: np.ndarray) -> np.ndarray:
        """Determine which region each point belongs to based on first feature.

        Args:
            points: Array of points (N, D) where first feature determines region.

        Returns:
            Array of region indices for each point.
        """
        if self._bounds is None:
            raise ValueError("Call generate() first to create region bounds")

        x0 = points[:, 0]
        regions = np.zeros(len(points), dtype=int)

        for i in range(len(self._bounds) - 1):
            if i < len(self._bounds) - 2:
                mask = (x0 >= self._bounds[i]) & (x0 < self._bounds[i + 1])
            else:
                mask = (x0 >= self._bounds[i]) & (x0 <= self._bounds[i + 1])
            regions[mask] = i

        return regions

    def sample_from_region(
        self, region_index: int, n_samples: int, n_inputs: int
    ) -> np.ndarray:
        """Sample random input points that fall within a specific region.

        The first feature is sampled uniformly within the region bounds,
        remaining features are sampled uniformly in [0, 1].

        Args:
            region_index: Which region to sample from.
            n_samples: Number of points to generate.
            n_inputs: Dimensionality of input space.

        Returns:
            Array of shape (n_samples, n_inputs).
        """
        low, high = self._region_bounds[region_index]
        x = np.random.rand(n_samples, n_inputs).astype(np.float32)
        x[:, 0] = low + (high - low) * x[:, 0]
        return x

    def get_constraint_indices_for_points(self, points: np.ndarray) -> np.ndarray:
        """Get constraint index for each point based on first feature.

        Args:
            points: Array of points (N, D).

        Returns:
            Array of constraint indices for each point.
        """

        regions = self.get_regions_for_points(points)
        return np.array([self._region_to_constraint[int(r)] for r in regions])

    def check(self, vectors: Any, inputs: np.ndarray) -> list[bool]:
        """Check if vectors satisfy their constraints.

        Args:
            vectors: Output vectors to check (N, n_outputs).
            inputs: Input vectors to determine which constraint applies (N, D).

        Returns:
            List of booleans indicating if each vector satisfies its constraint.
        """
        regions = self.get_regions_for_points(inputs)
        constraints = self.get_constraints_for_samples(regions)
        results = []
        for i in range(len(vectors)):
            results.append(self._check_one(vectors[i], inputs[i], constraints[i]))
        return results

    @abstractmethod
    def _check_one(self, vector: Any, input: np.ndarray, constraint: Any) -> bool:
        """Check if a single vector satisfies its constraint."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(c={len(self._constraints)}, regions={len(self._region_to_constraint)})"

    def _get_regions_for_constraint(
        self, constraint_idx: int
    ) -> list[tuple[int, float, float]]:
        """Get all regions that use a specific constraint.

        Returns:
            List of (region_idx, low_bound, high_bound) tuples.
        """
        regions = []
        for region_idx, c_idx in self._region_to_constraint.items():
            if c_idx == constraint_idx:
                low, high = self._region_bounds[region_idx]
                regions.append((region_idx, low, high))
        return sorted(regions)

    def to_str(self) -> dict[int, str]:
        """Get string representation of all constraint formulas with region bounds.

        Returns:
            Dict mapping constraint idx to string with bounds and formula.
        """
        result = ""
        for idx, c in self._constraints.items():
            regions = self._get_regions_for_constraint(idx)
            bounds_str = ", ".join(
                f"R{r}:[{low:.3f},{high:.3f}]" for r, low, high in regions
            )
            formula = self._to_str_one(c)
            result += f"\n[{bounds_str}]\n {formula}"
        return result

    @abstractmethod
    def _to_str_one(self, constraint: Any) -> str:
        """Convert a single constraint to string representation."""
        pass

    def _fix_one(self, vector: Any, input: np.ndarray, constraint: Any) -> Any:
        """Fix a single value to satisfy a constraint."""
        pass

    def fix(self, vectors, inputs: np.ndarray) -> list:
        regions = self.get_regions_for_points(inputs)
        constraints = self.get_constraints_for_samples(regions)
        already_valid = self.check(vectors, inputs)
        fixed_list = []
        for i in tqdm(range(len(vectors))):
            if already_valid[i]:
                fixed_list.append(vectors[i])
            else:
                fixed_list.append(self._fix_one(vectors[i], inputs[i], constraints[i]))
        return fixed_list

    def save_state(self, dir_path: str) -> None:
        """Persist the fully generated constraint so it can be reloaded later.

        Creates *dir_path* and writes:
          - ``constraint.json``  -- config + region mappings + human-readable
                                    formula strings
          - subclass-specific files (SMT-LIB2 or DIMACS) via
            ``_save_constraints``
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        self._save_constraints(dir_path)

        payload = {
            "class": type(self).__name__,
            "config": self.model_dump(),
            "state": {
                "region_to_constraint": {
                    str(k): v for k, v in self._region_to_constraint.items()
                },
                "region_bounds": {
                    str(k): list(v) for k, v in self._region_bounds.items()
                },
                "bounds": (self._bounds.tolist() if self._bounds is not None else None),
            },
            "formulas": self.to_str(),
        }
        with open(dir_path / "constraint.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)

    @staticmethod
    def load_state(dir_path: str) -> "Constraint":
        """Reconstruct a constraint from a directory written by ``save_state``.

        The Z3 / CNF formulas are deserialized from the subclass-specific
        files so that ``check``, ``fix``, etc. work without calling
        ``generate()`` again.
        """
        from nesy_data_gen.constraints.cnf import CNFConstraint
        from nesy_data_gen.constraints.smt_linear_inequality import (
            SMTLinearInequalityConstraint,
        )
        from nesy_data_gen.constraints.smt_region import SMTBoundedIntervalConstraint

        _CLASS_REGISTRY: dict[str, type] = {
            "CNFConstraint": CNFConstraint,
            "SMTBoundedIntervalConstraint": SMTBoundedIntervalConstraint,
            "SMTLinearInequalityConstraint": SMTLinearInequalityConstraint,
        }

        dir_path = Path(dir_path)
        with open(dir_path / "constraint.json") as f:
            data = json.load(f)

        cls_name = data["class"]
        if cls_name not in _CLASS_REGISTRY:
            raise ValueError(f"Unknown constraint class: {cls_name}")

        cls = _CLASS_REGISTRY[cls_name]
        obj = cls(**data["config"])

        state = data["state"]
        obj._region_to_constraint = {
            int(k): v for k, v in state["region_to_constraint"].items()
        }
        obj._region_bounds = {
            int(k): tuple[Any, ...](v) for k, v in state["region_bounds"].items()
        }
        obj._bounds = np.array(state["bounds"]) if state["bounds"] is not None else None

        obj._load_constraints(dir_path)
        return obj

    def _save_constraints(self, dir_path: Path) -> None:
        """Save constraint-specific data (override in subclasses)."""

    def _load_constraints(self, dir_path: Path) -> None:
        """Load constraint-specific data (override in subclasses)."""
