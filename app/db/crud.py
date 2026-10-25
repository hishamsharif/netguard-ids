import json
from typing import Any
import numpy as np
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import AlertLog, Feedback, FeatureLog, ThresholdLog, TrainingRun
from app.models.schemas import PredictResponse, FeedbackVerdict


async def insert_alert(db: AsyncSession, result: PredictResponse, raw_features: dict) -> int:
    row = AlertLog(
        rf_label=result.rf.label if result.rf else "SKIPPED",
        rf_confidence=result.rf.confidence if result.rf else 0.0,
        ae_score=result.ae.score if result.ae else 0.0,
        ae_threshold=result.ae.threshold if result.ae else 0.0,
        ae_label=result.ae.label if result.ae else "SKIPPED",
        consensus=result.consensus.value,
        severity=result.severity.value,
        raw_features=json.dumps(raw_features),
        rule_verdict=result.rule.verdict.value if result.rule else None,
        rule_triggered=json.dumps(result.rule.triggered_rules) if result.rule else None,
        ai_skipped=1 if result.ai_skipped else 0,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row.id


async def list_alerts(db: AsyncSession, limit: int = 100, offset: int = 0) -> list[AlertLog]:
    result = await db.execute(
        select(AlertLog).order_by(AlertLog.timestamp.desc()).limit(limit).offset(offset)
    )
    return result.scalars().all()


async def insert_feedback(
    db: AsyncSession, alert_id: int, verdict: FeedbackVerdict, notes: str | None
) -> Feedback:
    row = Feedback(alert_id=alert_id, verdict=verdict.value, notes=notes)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def log_threshold_change(
    db: AsyncSession, old: float, new: float, reason: str | None = None
) -> None:
    row = ThresholdLog(old_threshold=old, new_threshold=new, reason=reason)
    db.add(row)
    await db.commit()


async def insert_feature_log(
    db: AsyncSession, alert_id: int, features: "np.ndarray | list"
) -> None:
    feat_list = features.tolist() if hasattr(features, "tolist") else list(features)
    row = FeatureLog(
        alert_id=alert_id,
        features_json=json.dumps(feat_list),
    )
    db.add(row)
    await db.commit()


async def export_feedback_dataset(db: AsyncSession) -> list[dict[str, Any]]:
    """
    Feedback-label bridge: join FeatureLog + AlertLog + Feedback to produce
    corrected-label training rows.

    Label priority:
      FALSE_POSITIVE  → label="BENIGN"        (analyst corrected FP)
      TRUE_POSITIVE   → label=rf_label         (analyst confirmed attack class)
      FALSE_NEGATIVE  → label="UNKNOWN_ATTACK" (analyst found missed attack)
      No feedback     → label=rf_label only if rf_confidence >= 0.95 (auto-label)

    Returns list of {"features": [...], "label": str, "source": str, "alert_id": int}
    """
    # Rows with explicit analyst feedback
    feedback_rows = await db.execute(
        select(FeatureLog, AlertLog, Feedback)
        .join(AlertLog, FeatureLog.alert_id == AlertLog.id)
        .join(Feedback, Feedback.alert_id == AlertLog.id)
        .where(AlertLog.ai_skipped == 0)
    )
    records: list[dict] = []
    for fl, al, fb in feedback_rows:
        if fb.verdict == "FALSE_POSITIVE":
            label = "BENIGN"
        elif fb.verdict == "TRUE_POSITIVE":
            label = al.rf_label
        else:  # FALSE_NEGATIVE
            label = "UNKNOWN_ATTACK"
        records.append({
            "features": json.loads(fl.features_json),
            "label": label,
            "source": "feedback",
            "alert_id": al.id,
        })

    # High-confidence auto-labeled rows (no analyst feedback required)
    auto_rows = await db.execute(
        select(FeatureLog, AlertLog)
        .join(AlertLog, FeatureLog.alert_id == AlertLog.id)
        .where(AlertLog.ai_skipped == 0)
        .where(AlertLog.rf_confidence >= 0.95)
        .where(
            ~AlertLog.id.in_(
                select(Feedback.alert_id).scalar_subquery()
            )
        )
    )
    for fl, al in auto_rows:
        records.append({
            "features": json.loads(fl.features_json),
            "label": al.rf_label,
            "source": "auto",
            "alert_id": al.id,
        })

    return records


async def create_training_run(
    db: AsyncSession, stage: str, data_source: str
) -> int:
    row = TrainingRun(stage=stage, data_source=data_source, status="queued")
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row.id


async def update_training_run(
    db: AsyncSession,
    run_id: int,
    *,
    status: str | None = None,
    started_at=None,
    completed_at=None,
    metrics: dict | None = None,
    error: str | None = None,
) -> None:
    row = await db.get(TrainingRun, run_id)
    if not row:
        return
    if status:
        row.status = status
    if started_at:
        row.started_at = started_at
    if completed_at:
        row.completed_at = completed_at
    if metrics is not None:
        row.metrics_json = json.dumps(metrics)
    if error is not None:
        row.error = error
    await db.commit()


async def get_training_run(db: AsyncSession, run_id: int):
    return await db.get(TrainingRun, run_id)


async def list_training_runs(db: AsyncSession, limit: int = 20):
    result = await db.execute(
        select(TrainingRun).order_by(TrainingRun.created_at.desc()).limit(limit)
    )
    return result.scalars().all()


async def get_metrics_counts(db: AsyncSession) -> dict:
    total = await db.scalar(select(func.count()).select_from(AlertLog))
    feedback_count = await db.scalar(select(func.count()).select_from(Feedback))
    tp = await db.scalar(
        select(func.count()).select_from(Feedback).where(Feedback.verdict == "TRUE_POSITIVE")
    )
    fp = await db.scalar(
        select(func.count()).select_from(Feedback).where(Feedback.verdict == "FALSE_POSITIVE")
    )
    fn = await db.scalar(
        select(func.count()).select_from(Feedback).where(Feedback.verdict == "FALSE_NEGATIVE")
    )
    return {"total": total, "feedback_count": feedback_count, "tp": tp, "fp": fp, "fn": fn}
