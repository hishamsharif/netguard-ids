"""
TrainingPipeline — orchestrates the full end-to-end training workflow.

Stages:
  load → preprocess → train → evaluate → save artifacts

Each stage is interface-driven: swap any concrete implementation without
touching the pipeline itself. Config drives everything; reproducibility
is enforced by writing a config snapshot alongside every run's artifacts.

Usage:
  pipeline = TrainingPipeline.from_config(PipelineConfig(...))
  pipeline.run_rf()        # train + evaluate Random Forest
  pipeline.run_ae_phase1() # pretrain LSTM-AE on NSL-KDD
  pipeline.run_ae_phase2() # fine-tune LSTM-AE on CIC-IDS2017
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import joblib
import numpy as np

from training.interfaces import AbstractDataLoader, AbstractPreprocessor, EvalResults
from training.loaders.cicids2017_loader import CICIDS2017Loader
from training.loaders.nslkdd_loader import NSLKDDLoader
from training.preprocessors.cicids_preprocessor import CICIDSPreprocessor
from training.trainers.rf_trainer import RFTrainer, RFConfig
from training.trainers.ae_trainer import NSLKDDAETrainer, CICIDSAETrainer, AEConfig
from training.evaluators.classification_evaluator import ClassificationEvaluator, AnomalyEvaluator


@dataclass
class PipelineConfig:
    # Paths
    cicids_data_dir:  str = "data/cicids2017/"
    nslkdd_train_path: str = "data/nslkdd/KDDTrain+.txt"
    models_dir:       str = "models/"
    reports_dir:      str = "models/reports/"

    # Split
    test_size:    float = 0.15
    val_size:     float = 0.15
    random_state: int   = 42

    # RF hyperparameters
    rf: RFConfig = field(default_factory=RFConfig)

    # AE hyperparameters
    ae: AEConfig = field(default_factory=AEConfig)


class TrainingPipeline:

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        Path(config.models_dir).mkdir(parents=True, exist_ok=True)
        Path(config.reports_dir).mkdir(parents=True, exist_ok=True)
        self._save_config()

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "TrainingPipeline":
        return cls(config)

    # ── Public entry points ───────────────────────────────────────────

    def run_rf(self) -> EvalResults:
        """Full RF pipeline: load → preprocess → train → evaluate → save."""
        print("\n" + "="*60)
        print("  STAGE: Random Forest (CIC-IDS2017)")
        print("="*60)
        t0 = time.time()

        loader = CICIDS2017Loader()
        df = loader.load(self.cfg.cicids_data_dir)
        print(f"\nDataset overview:\n{json.dumps(loader.describe(df), indent=2)}")

        split = loader.split(
            df,
            test_size=self.cfg.test_size,
            val_size=self.cfg.val_size,
            random_state=self.cfg.random_state,
        )
        print(f"\nSplit — train:{split.meta['train_rows']:,}  "
              f"val:{split.meta['val_rows']:,}  test:{split.meta['test_rows']:,}")

        preprocessor = (
            CICIDSPreprocessor()
            .set_feature_names_in(split.feature_names)
        )
        X_train = preprocessor.fit_transform(split.X_train)
        X_val   = preprocessor.transform(split.X_val)
        X_test  = preprocessor.transform(split.X_test)

        trainer = RFTrainer(config=self.cfg.rf, label_encoder=loader.label_encoder)
        trainer.train(X_train, split.y_train, X_val, split.y_val)

        evaluator = ClassificationEvaluator()
        results = evaluator.evaluate(
            trainer.get_model(), X_test, split.y_test,
            label_names=split.label_names,
        )
        evaluator.print_summary(results)
        evaluator.save_report(results, self.cfg.reports_dir)

        # Save importances
        importances = trainer.feature_importances(preprocessor.get_feature_names_out())
        imp_path = Path(self.cfg.models_dir) / "rf_feature_importances.json"
        imp_path.write_text(json.dumps(importances[:30], indent=2))

        # Persist artifacts
        trainer.save(str(Path(self.cfg.models_dir) / "rf_cicids.joblib"))
        preprocessor.save(str(Path(self.cfg.models_dir) / "scaler.joblib"))
        feature_names_path = Path(self.cfg.models_dir) / "feature_names.json"
        feature_names_path.write_text(json.dumps(preprocessor.get_feature_names_out()))
        joblib.dump(loader.label_encoder, Path(self.cfg.models_dir) / "label_encoder.joblib")

        elapsed = time.time() - t0
        print(f"\n  RF pipeline done in {elapsed:.1f}s")
        return results

    def run_ae_phase1(self) -> dict:
        """Phase 1: pretrain LSTM-AE on NSL-KDD normal flows."""
        print("\n" + "="*60)
        print("  STAGE: LSTM-AE Phase 1 — NSL-KDD Pretrain")
        print("="*60)
        t0 = time.time()

        loader = NSLKDDLoader(normal_only=True)
        df = loader.load(self.cfg.nslkdd_train_path)
        split = loader.split(df, random_state=self.cfg.random_state)

        preprocessor = CICIDSPreprocessor().set_feature_names_in(split.feature_names)
        X_train = preprocessor.fit_transform(split.X_train)
        X_val   = preprocessor.transform(split.X_val)

        trainer = NSLKDDAETrainer(config=self.cfg.ae)
        trainer.train(X_train, None, X_val, None)
        trainer.save(str(Path(self.cfg.models_dir) / "lstm_ae_nslkdd.pt"))
        preprocessor.save(str(Path(self.cfg.models_dir) / "scaler_nslkdd.joblib"))

        elapsed = time.time() - t0
        print(f"\n  AE Phase 1 done in {elapsed:.1f}s  threshold={trainer.threshold:.6f}")
        return {"threshold": round(trainer.threshold, 6), "elapsed_s": round(elapsed, 1)}

    def run_ae_phase2(self) -> dict:
        """Phase 2: fine-tune LSTM-AE on CIC-IDS2017 benign flows."""
        print("\n" + "="*60)
        print("  STAGE: LSTM-AE Phase 2 — CIC-IDS2017 Fine-tune")
        print("="*60)
        t0 = time.time()

        loader = CICIDS2017Loader()
        df = loader.load(self.cfg.cicids_data_dir)

        # Use only benign flows for AE training (unsupervised normal-traffic model)
        label_col = loader.get_label_column()
        benign_df = df[df[label_col].str.upper() == "BENIGN"].copy()
        print(f"  Using {len(benign_df):,} benign flows for AE fine-tuning")

        split = loader.split(benign_df, random_state=self.cfg.random_state, stratify=False)

        # Re-use scaler from RF run if available; otherwise fit fresh
        scaler_path = Path(self.cfg.models_dir) / "scaler.joblib"
        if scaler_path.exists():
            preprocessor = CICIDSPreprocessor.load(str(scaler_path))
            X_train = preprocessor.transform(split.X_train)
            X_val   = preprocessor.transform(split.X_val)
            print("  Re-using RF scaler for consistency")
        else:
            preprocessor = CICIDSPreprocessor().set_feature_names_in(split.feature_names)
            X_train = preprocessor.fit_transform(split.X_train)
            X_val   = preprocessor.transform(split.X_val)
            preprocessor.save(str(scaler_path))

        # Clip to [-10, 10] — RobustScaler leaves IQR=0 features at raw scale
        # (e.g. sparse flag counts with values up to 10^6 produce trillion-scale MSE).
        AE_CLIP = 10.0
        X_train = np.clip(X_train, -AE_CLIP, AE_CLIP)
        X_val   = np.clip(X_val,   -AE_CLIP, AE_CLIP)
        print(f"  Clipped AE inputs to [{-AE_CLIP}, {AE_CLIP}] (handles IQR=0 features)")

        pretrained_path = str(Path(self.cfg.models_dir) / "lstm_ae_nslkdd.pt")
        trainer = CICIDSAETrainer(pretrained_path=pretrained_path, config=self.cfg.ae)
        trainer.train(X_train, None, X_val, None)
        trainer.save(str(Path(self.cfg.models_dir) / "lstm_ae_cicids.pt"))

        elapsed = time.time() - t0
        print(f"\n  AE Phase 2 done in {elapsed:.1f}s  threshold={trainer.threshold:.6f}")
        print(f"  → Update AE_THRESHOLD={trainer.threshold:.6f} in .env")
        return {"threshold": round(trainer.threshold, 6), "elapsed_s": round(elapsed, 1)}

    # ── Helpers ───────────────────────────────────────────────────────

    def _save_config(self) -> None:
        cfg_path = Path(self.cfg.models_dir) / "pipeline_config.json"

        def _serialise(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _serialise(v) for k, v in asdict(obj).items()}
            return obj

        cfg_path.write_text(json.dumps(_serialise(self.cfg), indent=2))
