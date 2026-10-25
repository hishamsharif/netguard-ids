"""
AbstractEvaluator — contract for model evaluation.

Evaluation is intentionally separate from training so different evaluators
can be plugged in without changing the trainer:
  - ClassificationEvaluator (v1): precision/recall/F1/confusion matrix
  - ROCEvaluator (v1.1): AUC-ROC per class
  - AnomalyEvaluator (v1): threshold calibration, reconstruction error dist
  - ComparativeEvaluator (v1.2): side-by-side model comparison
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class EvalResults:
    model_name: str
    metrics: dict                              # precision, recall, f1, etc.
    per_class: dict = field(default_factory=dict)  # per-label breakdown
    confusion_matrix: Any = None
    extra: dict = field(default_factory=dict)  # threshold dist, AUC, etc.


class AbstractEvaluator(ABC):

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        label_names: list[str] | None = None,
    ) -> EvalResults:
        """Run evaluation and return structured results."""

    @abstractmethod
    def print_summary(self, results: EvalResults) -> None:
        """Print a human-readable summary to stdout."""

    @abstractmethod
    def save_report(self, results: EvalResults, out_dir: str) -> None:
        """Persist the evaluation report (JSON + any plots) to out_dir."""
