"""Early stopping callback for SBBTS training."""

import copy
from typing import Optional

import torch.nn as nn


class EarlyStopping:
    """
    Halt training when validation loss stops improving, restore best weights.

    Args:
        patience: Epochs to wait without improvement before stopping
        min_delta: Minimum change to count as improvement
    """

    def __init__(self, patience: int = 50, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self._best_state: Optional[dict] = None
        self.triggered = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self._best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered

    def load_best_weights(self, model: nn.Module) -> None:
        if self._best_state is not None:
            model.load_state_dict(self._best_state)

    def reset(self) -> None:
        self.best_loss = float("inf")
        self.counter = 0
        self._best_state = None
        self.triggered = False
