"""
CLI entry point — Random Forest training.

Usage:
  python -m training.train_rf [--data DIR] [--trees N] [--jobs N]

All logic lives in training/pipelines/training_pipeline.py.
This script is intentionally thin: parse args, build config, delegate.
"""
import argparse
from training.trainers.rf_trainer import RFConfig
from training.pipelines.training_pipeline import PipelineConfig, TrainingPipeline


def main():
    p = argparse.ArgumentParser(description="Train Random Forest on CIC-IDS2017")
    p.add_argument("--data",     default="data/cicids2017/",   help="CIC-IDS2017 CSV directory")
    p.add_argument("--models",   default="models/",            help="Output directory for artifacts")
    p.add_argument("--reports",  default="models/reports/",    help="Output directory for eval reports")
    p.add_argument("--trees",    type=int, default=200,        help="Number of RF estimators")
    p.add_argument("--jobs",     type=int, default=-1,         help="Parallel jobs (-1 = all cores)")
    p.add_argument("--seed",     type=int, default=42,         help="Random seed")
    args = p.parse_args()

    config = PipelineConfig(
        cicids_data_dir=args.data,
        models_dir=args.models,
        reports_dir=args.reports,
        random_state=args.seed,
        rf=RFConfig(n_estimators=args.trees, n_jobs=args.jobs),
    )
    TrainingPipeline.from_config(config).run_rf()


if __name__ == "__main__":
    main()
