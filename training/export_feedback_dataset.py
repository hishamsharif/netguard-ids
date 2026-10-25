"""
Feedback-label bridge CLI.

Reads AlertLog + FeatureLog + Feedback from the live DB and exports a
corrected-label CSV suitable for incremental RF/AE retraining.

Label priority:
  FALSE_POSITIVE  → BENIGN
  TRUE_POSITIVE   → original rf_label (e.g. "DoS Hulk")
  FALSE_NEGATIVE  → UNKNOWN_ATTACK
  No feedback     → rf_label only when rf_confidence >= 0.95

Usage:
  python -m training.export_feedback_dataset [--db PATH] [--out PATH] [--min-conf 0.95]

Output CSV columns:
  feature_0 … feature_N-1, label, source, alert_id
"""
import argparse
import asyncio
import csv
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.db.models import AlertLog, FeatureLog, Feedback
from app.db.crud import export_feedback_dataset


async def _run(db_url: str, out_path: str, min_conf: float) -> None:
    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        records = await export_feedback_dataset(session)

    if not records:
        print("No records to export — check that FeatureLog has rows and Feedback exists.")
        return

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_features = len(records[0]["features"])
    fieldnames = [f"feature_{i}" for i in range(n_features)] + ["label", "source", "alert_id"]

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {f"feature_{i}": v for i, v in enumerate(rec["features"])}
            row["label"] = rec["label"]
            row["source"] = rec["source"]
            row["alert_id"] = rec["alert_id"]
            writer.writerow(row)

    feedback_count = sum(1 for r in records if r["source"] == "feedback")
    auto_count = sum(1 for r in records if r["source"] == "auto")
    print(f"Exported {len(records):,} rows → {out}")
    print(f"  feedback-corrected: {feedback_count:,}  auto-labeled (conf≥{min_conf}): {auto_count:,}")
    label_counts: dict[str, int] = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {label:<30} {count:>6,}")

    await engine.dispose()


def main() -> None:
    p = argparse.ArgumentParser(description="Export feedback-labeled dataset for retraining")
    p.add_argument("--db",       default="sqlite+aiosqlite:///netguard.db", help="Async DB URL")
    p.add_argument("--out",      default="data/feedback_dataset.csv",        help="Output CSV path")
    p.add_argument("--min-conf", type=float, default=0.95,                   help="Min RF confidence for auto-labels",
                   dest="min_conf")
    args = p.parse_args()
    asyncio.run(_run(args.db, args.out, args.min_conf))


if __name__ == "__main__":
    main()
