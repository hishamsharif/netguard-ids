import numpy as np
import torch
import joblib
from pathlib import Path
from app.config import settings
from app.models.lstm_ae import LSTMAutoencoder
from app.models.schemas import RFResult, AEResult


class RFInference:
    def __init__(self):
        self.model = None

    def load(self) -> None:
        if not Path(settings.rf_model_path).exists():
            raise FileNotFoundError(f"RF model not found: {settings.rf_model_path}")
        self.model = joblib.load(settings.rf_model_path)

    def predict(self, x: np.ndarray, top_k: int = 5) -> RFResult:
        label = self.model.predict(x)[0]
        proba = self.model.predict_proba(x)[0]
        confidence = float(proba.max())
        return RFResult(label=str(label), confidence=confidence, top_features=[])


class AEInference:
    def __init__(self):
        self.model: LSTMAutoencoder | None = None
        self.threshold: float = settings.ae_threshold

    def load(self) -> None:
        if not Path(settings.ae_model_path).exists():
            raise FileNotFoundError(f"AE model not found: {settings.ae_model_path}")
        self.model = LSTMAutoencoder.load(str(settings.ae_model_path))

    _CLIP = 10.0  # must match training_pipeline.AE_CLIP

    def predict(self, x: np.ndarray) -> AEResult:
        # x: [1, n_features] → clip (matches training) → [1, 1, n_features]
        x = np.clip(x, -self._CLIP, self._CLIP)
        tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        score = float(self.model.reconstruction_error(tensor)[0])
        label = "ANOMALY" if score > self.threshold else "BENIGN"
        return AEResult(score=score, threshold=self.threshold, label=label)

    def per_feature_error(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -self._CLIP, self._CLIP)
        tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        return self.model.per_feature_error(tensor)[0].numpy()


rf_inference = RFInference()
ae_inference = AEInference()
