"""Model1 trainer - simple feedforward network."""

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .base import BaseTrainer


class BaseDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Model1Network(nn.Module):
    """Simple feedforward network with optional sigmoid output and shield layer."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        use_sigmoid,
        shield_layer=None,
        dropout=0.0,
    ):
        super().__init__()
        self.num_layers = max(1, int(num_layers))
        self.use_sigmoid = use_sigmoid

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(self.num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.fc = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

        self.shield = shield_layer
        if shield_layer is not None:
            import inspect

            sig = inspect.signature(shield_layer.forward)
            self._shield_supports_goal = "goal" in sig.parameters
        else:
            self._shield_supports_goal = False

    def forward(self, inputs, goal=None, **kwargs):
        out = self.fc(inputs)
        if self.use_sigmoid:
            out = self.sigmoid(out)
        if self.shield is not None:
            pre_shield = out.clone().detach()
            if goal is not None and self._shield_supports_goal:
                out = self.shield(out, goal=goal)
            else:
                out = self.shield(out)
            if out.abs().max() > 100:
                print(f"[WARNING] Shield produced large values!")
                print(
                    f"  Pre-shield  min={pre_shield.min().item():.4f} max={pre_shield.max().item():.4f}"
                )
                print(
                    f"  Post-shield min={out.min().item():.4f} max={out.max().item():.4f}"
                )
                print(
                    f"  Input to model: {inputs[out.abs().max(dim=1).values > 100][0]}"
                )
                print(
                    f"  Pre-shield vals: {pre_shield[out.abs().max(dim=1).values > 100][0]}"
                )
                print(
                    f"  Post-shield vals: {out[out.abs().max(dim=1).values > 100][0]}"
                )

        return out


class Model1Trainer(BaseTrainer):
    """Trainer for Model1 using L1 loss."""

    def __init__(self, config: dict):
        self.config = config
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 100)
        self.lr = config.get("lr", 0.001)
        self.hidden_dim = config.get("hidden_dim", 64)
        self.num_layers = config.get("num_layers", 2)
        self.patience = config.get("patience", 10)
        self.device = config.get("device", "cpu")
        self.model = None
        self.is_real = config.get("is_real", False)
        self.dropout = config.get("dropout", 0.0)
        self.learning_seed = config.get("learning_seed", None)
        self.shield_layer = config.get("shield_layer", None)
        self.history = {"train_loss": [], "val_loss": []}
        if self.is_real:
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.BCELoss()

    def build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        return Model1Network(
            input_dim,
            output_dim,
            self.hidden_dim,
            self.num_layers,
            use_sigmoid=not self.is_real,
            shield_layer=self.shield_layer,
            dropout=self.dropout,
        )

    def compute_loss(self, outputs, targets, **kwargs) -> torch.Tensor:
        loss = self.criterion(outputs, targets)
        return loss

    def compute_semantic_loss(
        self, outputs, sl_module, targets=None, eps=1e-5
    ) -> torch.Tensor:
        constraint_prob = sl_module(outputs)
        if targets is None:
            semantic_loss = -torch.log(eps + constraint_prob)
        else:
            semantic_loss = -(
                targets * torch.log(eps + constraint_prob)
                + (1 - targets) * torch.log(eps + 1 - constraint_prob)
            )
        semantic_loss = semantic_loss.mean()
        return semantic_loss

    def train(self, x_train, y_train, x_val, y_val, constraint=None):
        """Train the model with validation and early stopping."""
        if self.learning_seed is not None:
            from nesy_data_gen.utils.seed import seed_everything

            seed_everything(self.learning_seed)

        input_dim = x_train.shape[1]
        output_dim = y_train.shape[1]

        if self.model is None:
            self.model = self.build_model(input_dim, output_dim).to(self.device)

        train_loader = DataLoader(
            BaseDataset(x_train, y_train), batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            BaseDataset(x_val, y_val), batch_size=self.batch_size, shuffle=False
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        self.history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                outputs = self.model(xb, goal=yb)
                loss = self.compute_loss(outputs, yb, constraint=constraint)

                if not torch.isnan(loss):
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
                    outputs = self.model(xb)
                    loss = self.compute_loss(outputs, yb, constraint=constraint)
                    if not torch.isnan(loss):
                        val_loss += loss.item() * len(xb)

            val_loss /= len(val_loader.dataset)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(
                f"Epoch [{epoch+1}/{self.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model (Val Loss: {best_val_loss:.4f})")

        return self.model

    def train_with_sl(
        self,
        x_train,
        y_neural_train,
        y_query_train,
        x_val,
        y_neural_val,
        y_query_val,
        sl_module,
        constraint=None,
    ):
        """Train the model with validation and early stopping."""
        if self.learning_seed is not None:
            from nesy_data_gen.utils.seed import seed_everything

            seed_everything(self.learning_seed)

        input_dim = x_train.shape[1]
        output_dim = y_neural_train.shape[1]

        if self.model is None:
            self.model = self.build_model(input_dim, output_dim).to(self.device)

        train_loader = DataLoader(
            BaseDataset(x_train, y_query_train),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            BaseDataset(x_val, y_query_val), batch_size=self.batch_size, shuffle=False
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                outputs = self.model(xb)
                loss = self.compute_semantic_loss(outputs, sl_module, yb)

                if not torch.isnan(loss):
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
                    outputs = self.model(xb)
                    loss = self.compute_semantic_loss(outputs, sl_module, yb)
                    if not torch.isnan(loss):
                        val_loss += loss.item() * len(xb)

            val_loss /= len(val_loader.dataset)

            print(
                f"Epoch [{epoch+1}/{self.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model (Val Loss: {best_val_loss:.4f})")

        return self.model

    def train_with_unsupervised_sl(
        self, x_train, y_train, x_val, y_val, x_unsup, sl_module, constraint=None
    ):
        """Train the model with validation and early stopping."""
        input_dim = x_train.shape[1]
        output_dim = y_train.shape[1]

        if self.model is None:
            self.model = self.build_model(input_dim, output_dim).to(self.device)

        train_loader = DataLoader(
            BaseDataset(x_train, y_train), batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            BaseDataset(x_val, y_val), batch_size=self.batch_size, shuffle=False
        )
        unsup_loader = DataLoader(
            BaseDataset(x_unsup, x_unsup), batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        semantic_loss_weight = self.config.get("semantic_loss_weight", 1.0)
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                outputs = self.model(xb)
                loss = self.compute_loss(outputs, yb, constraint=constraint)

                # semantic loss on unsupervised data
                xb_unsup, _ = next(iter(unsup_loader))
                xb_unsup = xb_unsup.to(self.device)
                outputs_unsup = self.model(xb_unsup)
                semantic_loss = self.compute_semantic_loss(outputs_unsup, sl_module)

                loss += semantic_loss_weight * semantic_loss

                if not torch.isnan(loss):
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
                    outputs = self.model(xb)
                    loss = self.compute_loss(outputs, yb, constraint=constraint)

                    # semantic loss on the val data
                    semantic_loss = self.compute_semantic_loss(outputs_unsup, sl_module)
                    loss += semantic_loss_weight * semantic_loss

                    if not torch.isnan(loss):
                        val_loss += loss.item() * len(xb)

            val_loss /= len(val_loader.dataset)

            print(
                f"Epoch [{epoch+1}/{self.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model (Val Loss: {best_val_loss:.4f})")

        return self.model

    def inference(self, x):
        """Run inference."""
        self.model.eval()
        x = x.to(self.device)
        self.model.to(self.device)

        with torch.no_grad():
            output = self.model(x)
        return output
