import json
from typing import Any
import numpy as np
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import AlertLog, Feedback, FeatureLog, ThresholdLog, TrainingRun, ModelVersion
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


async def seed_baseline_model_version(
    db: AsyncSession,
    metrics: dict,
    notes: str = "",
    models_ready: bool = True,
) -> None:
    """Insert the v1.0-baseline ModelVersion if none exists yet."""
    existing = await db.scalar(select(func.count()).select_from(ModelVersion))
    if existing and existing > 0:
        return
    row = ModelVersion(
        version_tag="v1.0-baseline" if models_ready else "v1.0-pending",
        stage="all",
        model_path_rf="models/rf_cicids.joblib" if models_ready else None,
        model_path_ae="models/lstm_ae_cicids.pt" if models_ready else None,
        is_active=1 if models_ready else 0,
        is_active_rf=1 if models_ready else 0,
        is_active_ae=1 if models_ready else 0,
        trained_on_source="csv",
        trained_up_to_alert_id=0,
        trained_flow_count=0,
        metrics_json=json.dumps(metrics) if metrics else None,
        training_run_id=None,
        notes=notes,
    )
    db.add(row)
    await db.commit()


async def get_active_model_version(db: AsyncSession) -> ModelVersion | None:
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.is_active == 1).order_by(ModelVersion.id.desc())
    )
    return result.scalars().first()


async def get_active_rf_version(db: AsyncSession) -> ModelVersion | None:
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.is_active_rf == 1).order_by(ModelVersion.id.desc())
    )
    return result.scalars().first()


async def get_active_ae_version(db: AsyncSession) -> ModelVersion | None:
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.is_active_ae == 1).order_by(ModelVersion.id.desc())
    )
    return result.scalars().first()


async def list_model_versions(db: AsyncSession) -> list[ModelVersion]:
    result = await db.execute(select(ModelVersion).order_by(ModelVersion.id.desc()))
    return result.scalars().all()


async def activate_model_version(db: AsyncSession, version_id: int) -> ModelVersion | None:
    """Deactivate all, activate the chosen version for all flags."""
    all_rows = await db.execute(select(ModelVersion))
    for v in all_rows.scalars().all():
        active = v.id == version_id
        v.is_active = 1 if active else 0
        v.is_active_rf = 1 if active else 0
        v.is_active_ae = 1 if active else 0
    await db.commit()
    return await db.get(ModelVersion, version_id)


async def activate_rf_version(db: AsyncSession, version_id: int) -> ModelVersion | None:
    """Switch which version's RF model is active — independent of AE."""
    all_rows = await db.execute(select(ModelVersion))
    for v in all_rows.scalars().all():
        v.is_active_rf = 1 if v.id == version_id else 0
    await db.commit()
    return await db.get(ModelVersion, version_id)


async def activate_ae_version(db: AsyncSession, version_id: int) -> ModelVersion | None:
    """Switch which version's AE model is active — independent of RF."""
    all_rows = await db.execute(select(ModelVersion))
    for v in all_rows.scalars().all():
        v.is_active_ae = 1 if v.id == version_id else 0
    await db.commit()
    return await db.get(ModelVersion, version_id)


async def count_pending_logs(db: AsyncSession, trained_up_to_alert_id: int) -> int:
    """Flows added since last DB training (eligible for incremental retraining)."""
    result = await db.scalar(
        select(func.count()).select_from(AlertLog)
        .where(AlertLog.id > trained_up_to_alert_id)
        .where(AlertLog.ai_skipped == 0)
    )
    return result or 0


async def register_trained_version(
    db: AsyncSession,
    *,
    version_tag: str,
    stage: str,
    training_run_id: int,
    model_path_rf: str | None,
    model_path_ae: str | None,
    trained_up_to_alert_id: int,
    trained_flow_count: int,
    metrics: dict,
) -> ModelVersion:
    """Deactivate previous active version, register and activate new one.
    is_active_rf / is_active_ae are only set for the stages actually trained."""
    trains_rf = stage in ("rf", "all")
    trains_ae = stage in ("ae", "ae1", "ae2", "all")

    # Clear unified active; clear stage-specific flags only for trained stages
    all_rows = (await db.execute(select(ModelVersion))).scalars().all()
    for v in all_rows:
        v.is_active = 0
        if trains_rf:
            v.is_active_rf = 0
        if trains_ae:
            v.is_active_ae = 0

    row = ModelVersion(
        version_tag=version_tag,
        stage=stage,
        model_path_rf=model_path_rf,
        model_path_ae=model_path_ae,
        is_active=1,
        is_active_rf=1 if trains_rf else 0,
        is_active_ae=1 if trains_ae else 0,
        trained_on_source="db",
        trained_up_to_alert_id=trained_up_to_alert_id,
        trained_flow_count=trained_flow_count,
        metrics_json=json.dumps(metrics),
        training_run_id=training_run_id,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


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
