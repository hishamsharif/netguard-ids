import numpy as np
import joblib
from pathlib import Path
from app.config import settings


class Preprocessor:
    def __init__(self):
        self.scaler = None
        self.selector = None
        self._feature_names = None
        self._loaded = False

    def load(self) -> None:
        if not Path(settings.scaler_path).exists():
            raise FileNotFoundError(f"Scaler not found: {settings.scaler_path}")
        raw = joblib.load(settings.scaler_path)
        # Training saves a dict {"pipeline": Pipeline, "feature_names_in": [...], ...}
        if isinstance(raw, dict):
            self.scaler = raw["pipeline"]
            self._feature_names = raw.get("feature_names_in", settings.feature_names)
        else:
            self.scaler = raw
            self._feature_names = settings.feature_names
        if Path(settings.selector_path).exists():
            self.selector = joblib.load(settings.selector_path)
        self._loaded = True

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names or []

    @property
    def training_medians(self) -> dict[str, float]:
        """Per-feature training medians from RobustScaler.center_ (for optional feature imputation)."""
        if not self._loaded or self.scaler is None:
            return {}
        try:
            sc = getattr(self.scaler, "named_steps", {}).get("scaler") or self.scaler
            if hasattr(sc, "center_") and self._feature_names:
                return dict(zip(self._feature_names, sc.center_.tolist()))
        except Exception:
            pass
        return {}

    def transform(self, features: dict[str, float]) -> np.ndarray:
        """Dict of raw CIC-IDS2017 features → scaled numpy array [1, n_features]."""
        if not self._loaded:
            raise RuntimeError("Preprocessor not loaded — call load() first")
        medians = self.training_medians
        ordered = [features.get(f, medians.get(f, 0.0)) for f in self._feature_names]
        x = np.array(ordered, dtype=np.float32).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.scaler.transform(x)
        if self.selector is not None:
            x = self.selector.transform(x)
        return x

preprocessor = Preprocessor()
