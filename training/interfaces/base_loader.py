"""
AbstractDataLoader — contract every dataset loader must satisfy.

A loader is responsible for:
  1. Reading raw source files into a DataFrame
  2. Declaring which column is the label
  3. Splitting into train / val / test with no leakage

Swapping to a new dataset (e.g., UNSW-NB15, KDD99) requires only a new
concrete class — nothing else in the pipeline changes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class SplitData:
    X_train: np.ndarray
    X_val:   np.ndarray
    X_test:  np.ndarray
    y_train: np.ndarray
    y_val:   np.ndarray
    y_test:  np.ndarray
    feature_names: list[str]
    label_names:   list[str]
    meta: dict = field(default_factory=dict)  # class distributions, row counts, etc.


class AbstractDataLoader(ABC):

    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """Read source files at `path` into a raw DataFrame. No preprocessing."""

    @abstractmethod
    def get_label_column(self) -> str:
        """Return the name of the target/label column as it appears in the raw DataFrame."""

    @abstractmethod
    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return ordered list of feature column names to use for training."""

    @abstractmethod
    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        stratify: bool = True,
    ) -> SplitData:
        """
        Stratified train/val/test split with no data leakage.
        Label encoding is performed here; the encoded integers are returned in SplitData.
        """

    def describe(self, df: pd.DataFrame) -> dict:
        """Optional: return class distribution and basic dataset statistics."""
        label_col = self.get_label_column()
        if label_col in df.columns:
            counts = df[label_col].value_counts().to_dict()
            return {"rows": len(df), "columns": len(df.columns), "class_distribution": counts}
        return {"rows": len(df), "columns": len(df.columns)}
