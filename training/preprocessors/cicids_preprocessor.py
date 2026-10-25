"""
CIC-IDS2017 Preprocessor — v1

Pipeline:
  1. VarianceThreshold  — drop near-constant features (variance < 1e-5)
  2. RobustScaler       — scale using median/IQR (outlier-resistant)

Deliberately deferred to v1.1:
  - SelectFromModel (RF importance-based feature selection)
  - SMOTE oversampling (needs training labels — goes in trainer, not preprocessor)

Why RobustScaler over StandardScaler:
  CIC-IDS2017 has extreme outliers (packet bursts, DoS attacks with
  millions of bytes/s). RobustScaler centres on median and scales on IQR,
  making it insensitive to these outliers while still normalising the range.
"""
from __future__ import annotations
import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from training.interfaces.base_preprocessor import AbstractPreprocessor


class CICIDSPreprocessor(AbstractPreprocessor):

    def __init__(self, variance_threshold: float = 1e-5):
        self.variance_threshold = variance_threshold
        self._pipeline: Pipeline | None = None
        self._feature_names_in: list[str] = []
        self._feature_names_out: list[str] = []
        self._fitted = False

    # ── AbstractPreprocessor interface ───────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "CICIDSPreprocessor":
        self._pipeline = Pipeline([
            ("variance", VarianceThreshold(threshold=self.variance_threshold)),
            ("scaler",   RobustScaler()),
        ])
        self._pipeline.fit(X)

        # Track which features survived variance filtering
        mask = self._pipeline.named_steps["variance"].get_support()
        self._feature_names_out = [
            self._feature_names_in[i] for i, keep in enumerate(mask) if keep
        ] if self._feature_names_in else [f"f{i}" for i in range(mask.sum())]

        n_dropped = (~mask).sum()
        if n_dropped:
            print(f"  VarianceThreshold dropped {n_dropped} near-constant features")
        print(f"  Features after preprocessing: {len(self._feature_names_out)}")

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        return self._pipeline.transform(X).astype(np.float32)

    def get_feature_names_out(self) -> list[str]:
        return list(self._feature_names_out)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self._pipeline,
            "feature_names_in": self._feature_names_in,
            "feature_names_out": self._feature_names_out,
            "variance_threshold": self.variance_threshold,
        }, path)
        print(f"  Preprocessor saved → {path}")

    @classmethod
    def load(cls, path: str) -> "CICIDSPreprocessor":
        data = joblib.load(path)
        obj = cls(variance_threshold=data["variance_threshold"])
        obj._pipeline = data["pipeline"]
        obj._feature_names_in = data["feature_names_in"]
        obj._feature_names_out = data["feature_names_out"]
        obj._fitted = True
        return obj

    # ── Extra helpers ─────────────────────────────────────────────────

    def set_feature_names_in(self, names: list[str]) -> "CICIDSPreprocessor":
        """Call before fit() to enable named output."""
        self._feature_names_in = list(names)
        return self

    @property
    def n_features_out(self) -> int:
        return len(self._feature_names_out)
