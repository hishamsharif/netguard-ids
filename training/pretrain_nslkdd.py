"""
CLI entry point — LSTM-AE Phase 1 (NSL-KDD pretrain).

Usage:
  python -m training.pretrain_nslkdd [--data FILE] [--models DIR] [--epochs N]
"""
import argparse
from training.trainers.ae_trainer import AEConfig
from training.pipelines.training_pipeline import PipelineConfig, TrainingPipeline


def main():
    p = argparse.ArgumentParser(description="Pretrain LSTM-AE on NSL-KDD normal flows")
    p.add_argument("--data",    default="data/nslkdd/KDDTrain+.txt", help="NSL-KDD training file")
    p.add_argument("--models",  default="models/",                   help="Output directory")
    p.add_argument("--epochs",  type=int, default=30,                help="Training epochs")
    p.add_argument("--batch",   type=int, default=512,               help="Batch size")
    p.add_argument("--hidden",  type=int, default=128,               help="LSTM hidden dim")
    p.add_argument("--seed",    type=int, default=42,                help="Random seed")
    args = p.parse_args()

    config = PipelineConfig(
        nslkdd_train_path=args.data,
        models_dir=args.models,
        random_state=args.seed,
        ae=AEConfig(epochs=args.epochs, batch_size=args.batch, hidden_dim=args.hidden),
    )
    TrainingPipeline.from_config(config).run_ae_phase1()


if __name__ == "__main__":
    main()
