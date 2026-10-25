"""
CLI entry point — LSTM-AE Phase 2 (CIC-IDS2017 fine-tune).

Requires Phase 1 artifact at models/lstm_ae_nslkdd.pt.

Usage:
  python -m training.finetune_cicids [--data DIR] [--models DIR] [--epochs N] [--unfreeze-epoch N]
"""
import argparse
from training.trainers.ae_trainer import AEConfig
from training.pipelines.training_pipeline import PipelineConfig, TrainingPipeline


def main():
    p = argparse.ArgumentParser(description="Fine-tune LSTM-AE on CIC-IDS2017 benign flows")
    p.add_argument("--data",           default="data/cicids2017/",  help="CIC-IDS2017 CSV directory")
    p.add_argument("--models",         default="models/",           help="Artifacts directory")
    p.add_argument("--epochs",         type=int, default=20,        help="Fine-tune epochs")
    p.add_argument("--unfreeze-epoch", type=int, default=10,        help="Epoch to unfreeze encoder",
                   dest="unfreeze_epoch")
    p.add_argument("--batch",          type=int, default=512,       help="Batch size")
    p.add_argument("--seed",           type=int, default=42,        help="Random seed")
    args = p.parse_args()

    config = PipelineConfig(
        cicids_data_dir=args.data,
        models_dir=args.models,
        random_state=args.seed,
        ae=AEConfig(
            epochs=args.epochs,
            batch_size=args.batch,
            unfreeze_epoch=args.unfreeze_epoch,
        ),
    )
    TrainingPipeline.from_config(config).run_ae_phase2()


if __name__ == "__main__":
    main()
