"""
AbstractTrainer — contract for all model training stages.

A trainer encapsulates one model type (RF, LSTM-AE, etc.) and the
hyperparameters/optimiser needed to train it.  It accepts pre-processed
numpy arrays and returns itself (the trained model lives inside it).

Concrete implementations:
  - RFTrainer      → scikit-learn RandomForestClassifier
  - AETrainer      → PyTorch LSTM Autoencoder (Phase 1: NSL-KDD pretrain)
  - TransferTrainer → Phase 2: fine-tune AE on CIC-IDS2017
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class TrainingHistory:
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class AbstractTrainer(ABC):

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray | None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "AbstractTrainer":
        """
        Train the model. Returns self so callers can chain .save().
        y_train is None for unsupervised models (AE).
        """

    @abstractmethod
    def get_model(self) -> Any:
        """Return the raw trained model object (sklearn estimator, nn.Module, etc.)."""

    @abstractmethod
    def get_history(self) -> TrainingHistory:
        """Return loss curves and any extra tracked metrics."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialise the trained model to disk."""
