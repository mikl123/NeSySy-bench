"""Sequential autoregressive model for multi-label classification."""

from __future__ import annotations

import copy
import itertools

import numpy as np
import torch
import torch.nn as nn
from pysat.solvers import Solver
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Partial feasibility check (SAT-based)
# ---------------------------------------------------------------------------


def _check_partial_feasibility(partial_vector: np.ndarray, constraint) -> bool:
    """Check if a partial assignment can still lead to a valid solution.

    Uses a SAT solver with assumptions only for decided positions (>= 0).
    Undecided positions (< 0) are left free for the solver to explore.
    Works with CNFConstraint — accesses the internal clause representation.
    """
    # Get the CNF clauses for region 0 (single-region case)
    cnf_obj = constraint._constraints[constraint._region_to_constraint[0]]
    clauses = cnf_obj._clauses

    assumptions = []
    for i, val in enumerate(partial_vector):
        if val < 0:
            continue  # undecided — leave free
        # DIMACS variables are 1-indexed
        assumptions.append((i + 1) if val >= 0.5 else -(i + 1))

    with Solver(bootstrap_with=clauses) as solver:
        return solver.solve(assumptions=assumptions) is True


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


class _SeqDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def generate_valuations(actuals: np.ndarray) -> np.ndarray:
    """Build sequential training rows from ground-truth label matrix.

    For each sample with labels [y0, y1, ..., y_{n-1}], creates n rows:
      step 0: partial = [-1, -1, ..., -1],       target = y0
      step 1: partial = [y0, -1, ..., -1],        target = y1
      ...
      step k: partial = [y0, ..., y_{k-1}, -1, ...], target = y_k

    Returns partial valuations of shape (n_samples * n_classes, n_classes - 1).
    """
    n_samples, n_classes = actuals.shape
    part_val_size = n_classes - 1
    partial_valuations = np.full(
        (n_samples * n_classes, part_val_size), -1, dtype=np.float32
    )
    for idx, actual in enumerate(actuals):
        for i in range(n_classes):
            row = idx * n_classes + i
            partial_valuations[row, :i] = actual[:i]
    return partial_valuations


def build_sequential_dataset(
    base_preds: np.ndarray,
    y: np.ndarray,
    x_orig: np.ndarray | None = None,
    with_initial: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Create (features, targets) for the sequential model.

    Parameters
    ----------
    base_preds : (n_samples, n_classes) base model predictions
    y : (n_samples, n_classes) ground-truth labels
    x_orig : (n_samples, n_input) original input features (optional)
    with_initial : if True, include base_preds in features

    Returns
    -------
    features : (n_samples * n_classes, feature_dim)
    targets : (n_samples * n_classes,)
    """
    n_samples, n_classes = y.shape
    partial = generate_valuations(y)

    # Build feature matrix
    parts = []
    if with_initial:
        parts.append(np.repeat(base_preds, n_classes, axis=0))
    if x_orig is not None:
        parts.append(np.repeat(x_orig, n_classes, axis=0))
    parts.append(partial)
    features = np.concatenate(parts, axis=1).astype(np.float32)

    # Targets: flatten y row-by-row
    targets = y.flatten().astype(np.float32)
    return features, targets


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SequentialModel(nn.Module):
    """Autoregressive model that predicts one label at a time.

    Input: [base_predictions, (optional: orig_input), partial_assignment]
    Output: scalar probability for the next label.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_size))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_size, 1))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc = nn.ModuleList(layers)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                x = self.sigmoid(self.fc[i](x))
            else:
                x = self.relu(self.fc[i](x))
                x = self.drop(x)
        return x.squeeze(-1)


class SequentialTrainer:
    """Train and run inference with a SequentialModel."""

    def __init__(
        self, model: SequentialModel | None, config: dict, input_dim: int | None = None
    ):
        self.config = config
        self.device = config.get("device", "cpu")
        self.batch_size = config.get("batch_size", 128)
        self.epochs = config.get("epochs", 500)
        self.lr = config.get("lr", 0.001)
        self.patience = config.get("patience", 20)
        self.learning_seed = config.get("learning_seed", None)
        self.criterion = nn.BCELoss()
        self.input_dim = input_dim
        self.model = model

    def _build_model(self, input_dim: int) -> SequentialModel:
        seq_config = {
            "hidden_size": 64,
            "num_layers": 3,
            "dropout": 0.1,
        }
        return SequentialModel(input_dim, **seq_config)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Train with early stopping on validation loss."""
        if self.learning_seed is not None:
            from nesy_data_gen.utils.seed import seed_everything

            seed_everything(self.learning_seed)
        if self.model is None:
            self.model = self._build_model(x_train.shape[1])
        self.model.to(self.device)
        train_loader = DataLoader(
            _SeqDataset(
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            _SeqDataset(
                torch.tensor(x_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = self.model(xb)
                loss = self.criterion(out, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_loader.dataset)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    out = self.model(xb)
                    loss = self.criterion(out, yb)
                    val_loss += loss.item() * len(xb)
            val_loss /= len(val_loader.dataset)

            print(
                f"[Sequential] Epoch [{epoch+1}/{self.epochs}] | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"[Sequential] Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"[Sequential] Loaded best model (Val Loss: {best_val_loss:.4f})")

    def greedy_predict(
        self,
        base_preds: np.ndarray,
        n_classes: int,
        x_orig: np.ndarray | None = None,
        with_initial: bool = True,
    ) -> np.ndarray:
        """Greedy autoregressive decoding.

        Returns (n_samples, n_classes) binary predictions.
        """
        self.model.eval()
        n_samples = len(base_preds)
        partial = np.full((n_samples, n_classes - 1), -1, dtype=np.float32)
        result = np.zeros((n_samples, n_classes), dtype=np.float32)

        for i in range(n_classes):
            parts = []
            if with_initial:
                parts.append(base_preds)
            if x_orig is not None:
                parts.append(x_orig)
            parts.append(partial)
            features = np.concatenate(parts, axis=1)

            dataset = _SeqDataset(
                torch.tensor(features, dtype=torch.float32),
                torch.zeros(n_samples, dtype=torch.float32),
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            probs = np.zeros(n_samples)
            cur = 0
            with torch.no_grad():
                for xb, _ in loader:
                    xb = xb.to(self.device)
                    p = self.model(xb).cpu().numpy()
                    probs[cur : cur + len(p)] = p
                    cur += len(p)

            preds = (probs > 0.5).astype(np.float32)
            result[:, i] = preds
            if i < n_classes - 1:
                partial[:, i] = preds

        return result

    def greedy_predict_with_fix(
        self,
        base_preds: np.ndarray,
        n_classes: int,
        constraint,
        x_inputs: np.ndarray,
        x_orig: np.ndarray | None = None,
        with_initial: bool = True,
    ) -> np.ndarray:
        """Greedy decoding with SAT-based constraint fixing at each step.

        At each step, the model predicts a probability. Uses a SAT solver
        to check if the partial assignment (only decided positions) is still
        feasible. If the greedy choice makes it infeasible, the alternative
        value is used.
        """
        self.model.eval()
        n_samples = len(base_preds)
        partial = np.full((n_samples, n_classes - 1), -1, dtype=np.float32)
        result = np.full((n_samples, n_classes), -1, dtype=np.float32)

        for i in range(n_classes):
            parts = []
            if with_initial:
                parts.append(base_preds)
            if x_orig is not None:
                parts.append(x_orig)
            parts.append(partial)
            features = np.concatenate(parts, axis=1)

            dataset = _SeqDataset(
                torch.tensor(features, dtype=torch.float32),
                torch.zeros(n_samples, dtype=torch.float32),
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            probs = np.zeros(n_samples)
            cur = 0
            with torch.no_grad():
                for xb, _ in loader:
                    xb = xb.to(self.device)
                    p = self.model(xb).cpu().numpy()
                    probs[cur : cur + len(p)] = p
                    cur += len(p)

            greedy = (probs > 0.5).astype(np.float32)

            # Check partial feasibility for each sample and flip if needed
            for s in range(n_samples):
                result[s, i] = greedy[s]
                if not _check_partial_feasibility(result[s], constraint):
                    alt = 1.0 - greedy[s]
                    result[s, i] = alt
                    if not _check_partial_feasibility(result[s], constraint):
                        raise RuntimeError(
                            f"Neither value is feasible at step {i} for sample {s}. "
                            f"Partial assignment: {result[s]}"
                        )

            if i < n_classes - 1:
                partial[:, i] = result[:, i]

        # Replace any remaining -1 with 0 (shouldn't happen)
        result[result < 0] = 0
        return result

    def beam_predict(
        self,
        base_preds: np.ndarray,
        n_classes: int,
        beam_k: int = 3,
        x_orig: np.ndarray | None = None,
        with_initial: bool = True,
    ) -> list[list[list]]:
        """Beam search decoding. Returns list of candidate lists per sample."""
        self.model.eval()
        n_samples = len(base_preds)
        # Each sample has a list of (combination, log_prob) candidates
        beams = [[[[-1] * (n_classes - 1), 1.0]] for _ in range(n_samples)]

        for i in range(n_classes):
            # Flatten all candidates for batched inference
            all_features = []
            shapes = []
            for s_idx in range(n_samples):
                shapes.append(len(beams[s_idx]))
                for comb, prob in beams[s_idx]:
                    parts = []
                    if with_initial:
                        parts.append(base_preds[s_idx])
                    if x_orig is not None:
                        parts.append(x_orig[s_idx])
                    parts.append(np.array(comb, dtype=np.float32))
                    all_features.append(np.concatenate(parts))

            features_t = torch.tensor(np.array(all_features), dtype=torch.float32)
            loader = DataLoader(
                _SeqDataset(features_t, torch.zeros(len(features_t))),
                batch_size=self.batch_size,
                shuffle=False,
            )
            probs = np.zeros(len(features_t))
            cur = 0
            with torch.no_grad():
                for xb, _ in loader:
                    xb = xb.to(self.device)
                    p = self.model(xb).cpu().numpy()
                    probs[cur : cur + len(p)] = p
                    cur += len(p)

            # Split probs back per sample and expand beams
            prob_idx = 0
            new_beams = []
            for s_idx in range(n_samples):
                candidates = []
                for b_idx in range(shapes[s_idx]):
                    comb, prev_prob = beams[s_idx][b_idx]
                    p = probs[prob_idx]
                    prob_idx += 1
                    for val, val_prob in [(1, p), (0, 1 - p)]:
                        new_comb = list(comb)
                        if i < n_classes - 1:
                            new_comb[i] = val
                        else:
                            new_comb.append(val)
                        candidates.append([new_comb, prev_prob * val_prob])
                # Keep top-k
                candidates.sort(key=lambda x: x[1], reverse=True)
                new_beams.append(candidates[:beam_k])
            beams = new_beams

        return beams


# ---------------------------------------------------------------------------
# Constraint-aware pseudo-labeling
# ---------------------------------------------------------------------------


def create_constraint_neg_examples(
    beam_results: list[list[list]],
    base_preds: np.ndarray,
    y_train: np.ndarray,
    constraint,
    x_orig: np.ndarray | None = None,
    with_initial: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate negative training examples from beam search constraint violations.

    For each beam candidate that violates the constraint, find the first
    position where it diverges from valid candidates, and create negative
    examples from that position onward with flipped targets.

    Returns (features, targets) to augment the sequential training set.
    """
    n_classes = y_train.shape[1]
    aug_features = []
    aug_targets = []

    for s_idx, candidates in enumerate(beam_results):
        # Separate valid and violating candidates
        valid = []
        violating = []
        for comb, prob in candidates:
            arr = np.array(comb[:n_classes], dtype=np.float32)
            if constraint.check(
                arr.reshape(1, -1), np.zeros((1, constraint.n_inputs), dtype=np.float32)
            )[0]:
                valid.append(comb[:n_classes])
            else:
                violating.append(comb[:n_classes])

        if not violating or not valid:
            continue

        for viol in violating:
            # Find first divergence position from any valid candidate
            min_div = n_classes
            for v in valid:
                for pos in range(n_classes):
                    if viol[pos] != v[pos]:
                        min_div = min(min_div, pos)
                        break
            if min_div >= n_classes:
                continue

            # Create negative examples from divergence point onward
            for pos in range(min_div, n_classes):
                partial = np.full(n_classes - 1, -1, dtype=np.float32)
                partial[:pos] = np.array(viol[:pos], dtype=np.float32)
                parts = []
                if with_initial:
                    parts.append(base_preds[s_idx])
                if x_orig is not None:
                    parts.append(x_orig[s_idx])
                parts.append(partial)
                aug_features.append(np.concatenate(parts))
                # Flipped target: opposite of the violating prediction
                aug_targets.append(1.0 - viol[pos])

    if not aug_features:
        return np.array([]), np.array([])
    return np.array(aug_features, dtype=np.float32), np.array(
        aug_targets, dtype=np.float32
    )
