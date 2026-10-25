"""
LSTM Autoencoder for network intrusion anomaly detection.

Architecture: Malhotra et al. (2016) "LSTM-based Encoder-Decoder for
Multi-sensor Anomaly Detection", adapted for CIC-IDS2017 flow features.

Transfer learning:
  Phase 1 — pretrain on NSL-KDD normal flows
  Phase 2 — fine-tune on CIC-IDS2017 normal flows (freeze encoder layers 0..N-1)
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, input_dim]
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h_n: torch.Tensor, c_n: torch.Tensor, seq_len: int):
        # Repeat context vector across time steps
        context = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)  # [B, T, hidden]
        out, _ = self.lstm(context, (h_n, c_n))
        return self.fc(out)  # [B, T, output_dim]


class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM Autoencoder.
    Anomaly score = mean per-feature reconstruction error over the window.

    Input:  [B, window_size, n_features]
    Output: [B, window_size, n_features]  (reconstruction)
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.encoder = LSTMEncoder(n_features, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(hidden_dim, n_features, num_layers, dropout)

    def forward(self, x: torch.Tensor):
        h_n, c_n = self.encoder(x)
        reconstruction = self.decoder(h_n, c_n, seq_len=x.size(1))
        return reconstruction

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error. Shape: [B]"""
        with torch.no_grad():
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=(1, 2))

    def per_feature_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-feature mean error over time window. Shape: [B, n_features]"""
        with torch.no_grad():
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=1)

    def freeze_encoder(self) -> None:
        """Freeze encoder weights for transfer learning fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "LSTMAutoencoder":
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            n_features=checkpoint["n_features"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def save(self, path: str) -> None:
        torch.save(
            {
                "n_features": self.n_features,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.encoder.lstm.num_layers,
                "state_dict": self.state_dict(),
            },
            path,
        )
