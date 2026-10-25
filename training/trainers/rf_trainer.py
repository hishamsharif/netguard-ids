"""
Random Forest Trainer — v1

Design decisions for CIC-IDS2017:
  - class_weight="balanced_subsample": each tree's bootstrap sample is
    rebalanced, preventing majority-class dominance without SMOTE overhead.
  - n_estimators=200: enough for stable feature importances on 79 features.
  - min_samples_leaf=5: prevents overfitting on very small attack classes.
  - random_state=42: reproducibility.

Deferred to v1.1:
  - SelectFromModel feature pruning after initial fit
  - GridSearchCV hyperparameter tuning
  - SMOTE integration
"""
from __future__ import annotations
import json
import joblib
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from training.interfaces.base_trainer import AbstractTrainer, TrainingHistory


@dataclass
class RFConfig:
    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_leaf: int = 5
    class_weight: str = "balanced_subsample"
    n_jobs: int = -1
    random_state: int = 42
    verbose: int = 1


class RFTrainer(AbstractTrainer):

    def __init__(self, config: RFConfig | None = None, label_encoder: LabelEncoder | None = None):
        self.config = config or RFConfig()
        self.label_encoder = label_encoder
        self._model: RandomForestClassifier | None = None
        self._history = TrainingHistory()

    # ── AbstractTrainer interface ─────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "RFTrainer":
        cfg = self.config
        print(f"Training Random Forest: {cfg.n_estimators} trees, "
              f"{X_train.shape[1]} features, {len(X_train):,} rows")

        self._model = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            class_weight=cfg.class_weight,
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
            verbose=cfg.verbose,
        )
        self._model.fit(X_train, y_train)
        self._history.extra["oob_score"] = None  # oob_score not enabled by default (cost)
        return self

    def get_model(self) -> RandomForestClassifier:
        if self._model is None:
            raise RuntimeError("Model not trained yet")
        return self._model

    def get_history(self) -> TrainingHistory:
        return self._history

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save untrained model")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        print(f"  RF model saved → {path}")

        if self.label_encoder is not None:
            le_path = str(path).replace(".joblib", "_le.joblib")
            joblib.dump(self.label_encoder, le_path)
            print(f"  Label encoder saved → {le_path}")

    # ── Convenience ───────────────────────────────────────────────────

    def feature_importances(self, feature_names: list[str]) -> list[dict]:
        """Return top features sorted by Gini importance."""
        if self._model is None:
            raise RuntimeError("Model not trained yet")
        imps = self._model.feature_importances_
        return sorted(
            [{"feature": n, "importance": float(v)} for n, v in zip(feature_names, imps)],
            key=lambda d: d["importance"],
            reverse=True,
        )
