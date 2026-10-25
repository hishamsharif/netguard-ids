import numpy as np
import joblib
from pathlib import Path
from app.config import settings


class Preprocessor:
    def __init__(self):
        self.scaler = None
        self.selector = None
        self._loaded = False

    def load(self) -> None:
        if not Path(settings.scaler_path).exists():
            raise FileNotFoundError(f"Scaler not found: {settings.scaler_path}")
        self.scaler = joblib.load(settings.scaler_path)
        if Path(settings.selector_path).exists():
            self.selector = joblib.load(settings.selector_path)
        self._loaded = True

    def transform(self, features: dict[str, float]) -> np.ndarray:
        """Dict of raw CIC-IDS2017 features → scaled numpy array [1, n_features]."""
        if not self._loaded:
            raise RuntimeError("Preprocessor not loaded — call load() first")
        ordered = [features.get(f, 0.0) for f in settings.feature_names]
        x = np.array(ordered, dtype=np.float32).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.scaler.transform(x)
        if self.selector is not None:
            x = self.selector.transform(x)
        return x

preprocessor = Preprocessor()
