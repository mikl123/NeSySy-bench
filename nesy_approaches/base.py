"""Base class for model trainers."""

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseTrainer(ABC):
    """Abstract base class for model trainers."""

    @abstractmethod
    def __init__(self, config: dict):
        pass

    @abstractmethod
    def build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        pass

    @abstractmethod
    def compute_loss(self, outputs, targets, **kwargs):
        pass

    @abstractmethod
    def train(self, x_train, y_train, x_val, y_val, constraint=None):
        pass

    @abstractmethod
    def inference(self, x):
        pass
