"""
LSTM Autoencoder Trainer — v1

Two concrete trainers in this module:
  NSLKDDAETrainer   — Phase 1: pretrain on NSL-KDD normal flows
  CICIDSAETrainer   — Phase 2: fine-tune on CIC-IDS2017 normal flows (transfer learning)

Transfer learning strategy (Malhotra et al., 2016):
  1. Pretrain encoder+decoder on NSL-KDD (smaller, well-cited IDS benchmark)
     to learn general normal-traffic representations.
  2. Load pretrained weights, FREEZE encoder layers.
  3. Fine-tune only the decoder on CIC-IDS2017 normal flows to adapt to
     the modern, higher-dimensional traffic feature space.
  4. After N epochs, UNFREEZE encoder for joint fine-tuning at lower LR.
  5. Threshold θ = 95th percentile of reconstruction error on normal val set.

Why this order:
  - The encoder has learned "what normal traffic looks like" from NSL-KDD.
  - Fine-tuning decoder only adapts the reconstruction head to CIC-IDS2017
    without catastrophic forgetting of the encoder's representations.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from app.models.lstm_ae import LSTMAutoencoder
from training.interfaces.base_trainer import AbstractTrainer, TrainingHistory


@dataclass
class AEConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 512
    epochs: int = 30
    lr: float = 1e-3
    clip_grad_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Transfer learning (Phase 2 only)
    unfreeze_epoch: int = 10
    finetune_lr: float = 1e-4


class _BaseAETrainer(AbstractTrainer):
    """Shared training loop for both Phase 1 and Phase 2."""

    def __init__(self, config: AEConfig | None = None):
        self.config = config or AEConfig()
        self._model: LSTMAutoencoder | None = None
        self._history = TrainingHistory()
        self._threshold: float = 0.023  # updated after threshold calibration

    def _run_epoch(
        self,
        model: LSTMAutoencoder,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train: bool,
    ) -> float:
        model.train() if train else model.eval()
        total_loss, total_n = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for (batch,) in loader:
                batch = batch.to(self.config.device)
                if train:
                    optimizer.zero_grad()
                recon = model(batch)
                loss = criterion(recon, batch)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.clip_grad_norm
                    )
                    optimizer.step()
                total_loss += loss.item() * len(batch)
                total_n += len(batch)
        return total_loss / max(total_n, 1)

    def _calibrate_threshold(
        self, model: LSTMAutoencoder, loader: DataLoader, percentile: float = 95.0
    ) -> float:
        model.eval()
        scores = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.config.device)
                scores.extend(model.reconstruction_error(batch).cpu().tolist())
        threshold = float(np.percentile(scores, percentile))
        print(f"  AE threshold (p{percentile:.0f} of normal scores): {threshold:.6f}")
        return threshold

    def _to_loader(self, X: np.ndarray, shuffle: bool = True) -> DataLoader:
        # Reshape to [N, 1, F] — window_size=1 per flow
        tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        return DataLoader(
            TensorDataset(tensor),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
        )

    def get_model(self) -> LSTMAutoencoder:
        if self._model is None:
            raise RuntimeError("Model not trained yet")
        return self._model

    def get_history(self) -> TrainingHistory:
        return self._history

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save untrained model")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save(path)
        print(f"  AE model saved → {path}")

    @property
    def threshold(self) -> float:
        return self._threshold


class NSLKDDAETrainer(_BaseAETrainer):
    """
    Phase 1: Train LSTM-AE from scratch on NSL-KDD normal flows.
    Input features are NSL-KDD's 41 numerical features (after ordinal encoding).
    """

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "NSLKDDAETrainer":
        cfg = self.config
        n_features = X_train.shape[1]
        print(f"[Phase 1] Pretraining LSTM-AE on NSL-KDD: {n_features} features, "
              f"{len(X_train):,} normal flows, device={cfg.device}")

        self._model = LSTMAutoencoder(
            n_features=n_features,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(cfg.device)

        train_loader = self._to_loader(X_train, shuffle=True)
        val_loader = self._to_loader(X_val, shuffle=False) if X_val is not None else None

        optimizer = torch.optim.Adam(self._model.parameters(), lr=cfg.lr)
        criterion = nn.MSELoss()

        for epoch in range(1, cfg.epochs + 1):
            train_loss = self._run_epoch(self._model, train_loader, optimizer, criterion, train=True)
            self._history.epochs.append(epoch)
            self._history.train_loss.append(train_loss)

            val_info = ""
            if val_loader:
                val_loss = self._run_epoch(self._model, val_loader, optimizer, criterion, train=False)
                self._history.val_loss.append(val_loss)
                val_info = f"  val_loss={val_loss:.6f}"

            print(f"  Epoch {epoch:3d}/{cfg.epochs}  train_loss={train_loss:.6f}{val_info}")

        # Calibrate threshold on validation set
        if val_loader:
            self._threshold = self._calibrate_threshold(self._model, val_loader)
        else:
            self._threshold = self._calibrate_threshold(self._model, train_loader)

        self._history.extra["threshold"] = self._threshold
        return self


class CICIDSAETrainer(_BaseAETrainer):
    """
    Phase 2: Fine-tune pretrained LSTM-AE on CIC-IDS2017 normal flows.
    Encoder frozen for first `unfreeze_epoch` epochs; joint fine-tuning after.
    """

    def __init__(
        self,
        pretrained_path: str,
        config: AEConfig | None = None,
    ):
        super().__init__(config)
        self._pretrained_path = pretrained_path

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "CICIDSAETrainer":
        cfg = self.config
        n_features = X_train.shape[1]
        print(f"[Phase 2] Fine-tuning LSTM-AE on CIC-IDS2017 benign flows: "
              f"{n_features} features, {len(X_train):,} rows")

        # Load pretrained checkpoint for weight transfer
        ckpt = torch.load(self._pretrained_path, map_location=cfg.device)
        pretrained_n_features = ckpt["n_features"]

        # Build a fresh model sized for the new dataset
        self._model = LSTMAutoencoder(
            n_features=n_features,
            hidden_dim=ckpt.get("hidden_dim", cfg.hidden_dim),
            num_layers=ckpt.get("num_layers", cfg.num_layers),
            dropout=cfg.dropout,
        ).to(cfg.device)

        if pretrained_n_features != n_features:
            # Partial weight transfer: copy only weights whose shape is unchanged.
            # Transferable: encoder.lstm.weight_hh_l* (hidden→hidden, [4H,H]),
            #               all bias terms ([4H]), full decoder LSTM (hidden→hidden).
            # Not transferable: encoder.lstm.weight_ih_l0 ([4H, F_old] vs [4H, F_new]),
            #                   decoder.fc.weight/bias ([F_old,H] vs [F_new,H]).
            print(f"  Feature dim change {pretrained_n_features}→{n_features}: "
                  f"partial weight transfer (hh-weights + biases + decoder LSTM)")
            pretrained_sd = ckpt["state_dict"]
            new_sd = self._model.state_dict()
            copied, skipped = 0, 0
            for key, val in pretrained_sd.items():
                if key in new_sd and new_sd[key].shape == val.shape:
                    new_sd[key] = val
                    copied += 1
                else:
                    skipped += 1
            self._model.load_state_dict(new_sd)
            print(f"  Copied {copied} weight tensors, skipped {skipped} (dim mismatch)")
        else:
            self._model.load_state_dict(ckpt["state_dict"])
            print(f"  Full weight transfer (same feature dim)")

        self._model.freeze_encoder()
        print(f"  Encoder frozen for first {cfg.unfreeze_epoch} epochs")

        train_loader = self._to_loader(X_train, shuffle=True)
        val_loader = self._to_loader(X_val, shuffle=False) if X_val is not None else None

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self._model.parameters()), lr=cfg.lr
        )
        criterion = nn.MSELoss()

        for epoch in range(1, cfg.epochs + 1):
            if epoch == cfg.unfreeze_epoch:
                self._model.unfreeze_encoder()
                optimizer = torch.optim.Adam(self._model.parameters(), lr=cfg.finetune_lr)
                print(f"  Epoch {epoch}: encoder unfrozen, lr → {cfg.finetune_lr}")

            train_loss = self._run_epoch(self._model, train_loader, optimizer, criterion, train=True)
            self._history.epochs.append(epoch)
            self._history.train_loss.append(train_loss)

            val_info = ""
            if val_loader:
                val_loss = self._run_epoch(self._model, val_loader, optimizer, criterion, train=False)
                self._history.val_loss.append(val_loss)
                val_info = f"  val_loss={val_loss:.6f}"

            print(f"  Epoch {epoch:3d}/{cfg.epochs}  train_loss={train_loss:.6f}{val_info}")

        # Threshold: use validation set (no label leakage — val is normal flows only)
        calibration_loader = val_loader or train_loader
        self._threshold = self._calibrate_threshold(self._model, calibration_loader)
        self._history.extra["threshold"] = self._threshold
        print(f"  → Set AE_THRESHOLD={self._threshold:.6f} in .env")
        return self
