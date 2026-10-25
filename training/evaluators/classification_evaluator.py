"""
Classification Evaluator — v1

Covers supervised (RF) and unsupervised (AE) evaluation:

  ClassificationEvaluator  — for RF: precision/recall/F1/confusion matrix
  AnomalyEvaluator         — for AE: reconstruction error dist, threshold stats

Deferred to v1.1:
  - ROC/AUC curves per class
  - SHAP-based post-hoc feature importance in evaluation
  - Learning curves (train size vs. F1)

Why macro-F1 as the headline metric:
  CIC-IDS2017 is class-imbalanced (~80% benign). Macro-F1 averages F1 across
  all classes equally, giving rare attack types equal weight. Weighted-F1
  would be inflated by the dominant benign class and mask poor recall on
  minority attack types like Infiltration or Heartbleed.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Any
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
)

from training.interfaces.base_evaluator import AbstractEvaluator, EvalResults


class ClassificationEvaluator(AbstractEvaluator):
    """Evaluates an sklearn classifier (predict + predict_proba)."""

    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        label_names: list[str] | None = None,
    ) -> EvalResults:
        y_pred = model.predict(X_test)

        # Headline metrics (macro-averaged — equal weight per class)
        metrics = {
            "accuracy":          float(accuracy_score(y_test, y_pred)),
            "f1_macro":          float(f1_score(y_test, y_pred, average="macro",    zero_division=0)),
            "f1_weighted":       float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "precision_macro":   float(precision_score(y_test, y_pred, average="macro",    zero_division=0)),
            "recall_macro":      float(recall_score(y_test, y_pred, average="macro",       zero_division=0)),
        }

        # Per-class breakdown
        report = classification_report(
            y_test, y_pred,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
        per_class = {
            k: v for k, v in report.items()
            if isinstance(v, dict)  # skip "accuracy", "macro avg", "weighted avg" scalars
        }

        cm = confusion_matrix(y_test, y_pred)

        return EvalResults(
            model_name="RandomForest",
            metrics=metrics,
            per_class=per_class,
            confusion_matrix=cm,
        )

    def print_summary(self, results: EvalResults) -> None:
        print(f"\n{'='*60}")
        print(f"  {results.model_name} Evaluation")
        print(f"{'='*60}")
        m = results.metrics
        print(f"  Accuracy:         {m['accuracy']:.4f}")
        print(f"  F1 (macro):       {m['f1_macro']:.4f}")
        print(f"  F1 (weighted):    {m['f1_weighted']:.4f}")
        print(f"  Precision (macro):{m['precision_macro']:.4f}")
        print(f"  Recall (macro):   {m['recall_macro']:.4f}")
        print(f"\n  Per-class F1:")
        for cls, stats in sorted(results.per_class.items()):
            if isinstance(stats, dict):
                print(f"    {cls:<30} f1={stats.get('f1-score', 0):.3f}  "
                      f"support={int(stats.get('support', 0)):,}")
        print(f"{'='*60}\n")

    def save_report(self, results: EvalResults, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        report_path = Path(out_dir) / "eval_rf.json"
        payload = {
            "model_name": results.model_name,
            "metrics": results.metrics,
            "per_class": results.per_class,
            "confusion_matrix": results.confusion_matrix.tolist()
                if results.confusion_matrix is not None else None,
        }
        report_path.write_text(json.dumps(payload, indent=2))
        print(f"  Evaluation report saved → {report_path}")


class AnomalyEvaluator(AbstractEvaluator):
    """
    Evaluates an LSTM-AE by treating it as a binary anomaly detector.
    Expects y_test to be binary: 0 = normal, 1 = anomaly.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        label_names: list[str] | None = None,
    ) -> EvalResults:
        import torch
        tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        scores = model.reconstruction_error(tensor).numpy()
        y_pred = (scores > self.threshold).astype(int)

        metrics = {
            "f1_macro":        float(f1_score(y_test, y_pred, average="macro",    zero_division=0)),
            "f1_weighted":     float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro",    zero_division=0)),
            "recall_macro":    float(recall_score(y_test, y_pred, average="macro",       zero_division=0)),
            "accuracy":        float(accuracy_score(y_test, y_pred)),
            "threshold":       self.threshold,
            "mean_recon_error_normal":  float(scores[y_test == 0].mean()) if (y_test == 0).any() else None,
            "mean_recon_error_anomaly": float(scores[y_test == 1].mean()) if (y_test == 1).any() else None,
        }

        cm = confusion_matrix(y_test, y_pred)
        return EvalResults(
            model_name="LSTM-AE",
            metrics=metrics,
            confusion_matrix=cm,
            extra={"reconstruction_scores": scores.tolist()[:500]},  # sample for report
        )

    def print_summary(self, results: EvalResults) -> None:
        print(f"\n{'='*60}")
        print(f"  {results.model_name} Anomaly Evaluation  (threshold={results.metrics['threshold']:.6f})")
        print(f"{'='*60}")
        m = results.metrics
        print(f"  Accuracy:          {m['accuracy']:.4f}")
        print(f"  F1 (macro):        {m['f1_macro']:.4f}")
        print(f"  Precision (macro): {m['precision_macro']:.4f}")
        print(f"  Recall (macro):    {m['recall_macro']:.4f}")
        if m.get("mean_recon_error_normal") is not None:
            print(f"  Mean recon error (normal):  {m['mean_recon_error_normal']:.6f}")
        if m.get("mean_recon_error_anomaly") is not None:
            print(f"  Mean recon error (anomaly): {m['mean_recon_error_anomaly']:.6f}")
        print(f"{'='*60}\n")

    def save_report(self, results: EvalResults, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        report_path = Path(out_dir) / "eval_ae.json"
        payload = {
            "model_name": results.model_name,
            "metrics": results.metrics,
            "confusion_matrix": results.confusion_matrix.tolist()
                if results.confusion_matrix is not None else None,
        }
        report_path.write_text(json.dumps(payload, indent=2))
        print(f"  Evaluation report saved → {report_path}")
