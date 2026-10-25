"""
Explainer — lightweight feature importance without SHAP.

RF explanation uses global model.feature_importances_ ranked for the predicted
class (fast; no C++ extension required).  AE explanation uses per-feature
reconstruction error (per-sample, meaningful for anomaly scoring).

Full per-sample SHAP wired to stored feature_log is queued as FE-006.
"""
import numpy as np
from app.services.inference import rf_inference, ae_inference
from app.config import settings


class Explainer:
    def load(self) -> None:
        pass  # nothing to load; uses live model references

    def rf_importance(self, top_k: int = 10) -> list[dict]:
        """Global RF feature importances — top-k by Gini importance."""
        if rf_inference.model is None:
            return []
        importances = rf_inference.model.feature_importances_
        names = settings.feature_names or [f"f{i}" for i in range(len(importances))]
        pairs = sorted(zip(names, importances), key=lambda t: t[1], reverse=True)
        return [{"feature": n, "importance": round(float(v), 6)} for n, v in pairs[:top_k]]

    def ae_feature_error(self, x: np.ndarray, top_k: int = 10) -> list[dict]:
        """Per-sample AE reconstruction error per feature."""
        errors = ae_inference.per_feature_error(x)
        names = settings.feature_names or [f"f{i}" for i in range(len(errors))]
        pairs = sorted(zip(names, errors), key=lambda t: t[1], reverse=True)
        return [{"feature": n, "error": round(float(v), 6)} for n, v in pairs[:top_k]]

    def natural_language(
        self, rf_importance: list[dict], ae_errors: list[dict], consensus: str
    ) -> str:
        top_rf = ", ".join(d["feature"] for d in rf_importance[:3])
        top_ae = ", ".join(d["feature"] for d in ae_errors[:3])
        return (
            f"Consensus: {consensus}. "
            f"Top RF features by global importance: {top_rf or 'unavailable'}. "
            f"Highest AE reconstruction error in: {top_ae or 'unavailable'}."
        )


explainer = Explainer()
