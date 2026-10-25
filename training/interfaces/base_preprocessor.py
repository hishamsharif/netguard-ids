"""
AbstractPreprocessor — contract for all feature-engineering / scaling stages.

A preprocessor is always fit on training data and applied to val/test without
re-fitting (to prevent leakage). It must be serialisable so the inference
service can load the same artifact at serving time.

Concrete implementations can add:
  - RobustScaler / MinMaxScaler / StandardScaler
  - VarianceThreshold (drop near-constant features)
  - SelectFromModel (RF importance-based selection)
  - PCA / UMAP (dimensionality reduction)
  - SMOTE oversampling (training set only)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class AbstractPreprocessor(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AbstractPreprocessor":
        """Learn parameters from training data. Returns self for chaining."""

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply learned transformation. Must not re-fit."""

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Convenience: fit then transform in one call."""
        return self.fit(X, y).transform(X)

    @abstractmethod
    def get_feature_names_out(self) -> list[str]:
        """Return the feature names after transformation (post selection/projection)."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist to disk. Must be loadable by the inference Preprocessor."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "AbstractPreprocessor":
        """Restore from disk."""
