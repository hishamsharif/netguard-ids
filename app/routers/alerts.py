from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.db.models import AlertLog, Feedback
from app.models.schemas import AlertRow, FeedbackRequest
from app.db.crud import list_alerts, insert_feedback
import json

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("/{alert_id}/detail")
async def get_alert_detail(alert_id: int, db: AsyncSession = Depends(get_db)):
    """Return full alert including stored raw_features for explain-on-demand."""
    result = await db.execute(select(AlertLog).where(AlertLog.id == alert_id))
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")
    fb_result = await db.execute(select(Feedback).where(Feedback.alert_id == alert_id))
    feedbacks = fb_result.scalars().all()
    return {
        "id": row.id,
        "timestamp": row.timestamp,
        "rf_label": row.rf_label,
        "rf_conf": row.rf_confidence,
        "ae_score": row.ae_score,
        "ae_threshold": row.ae_threshold,
        "ae_label": row.ae_label,
        "consensus": row.consensus,
        "severity": row.severity,
        "rule_verdict": row.rule_verdict,
        "rule_triggered": json.loads(row.rule_triggered) if row.rule_triggered else [],
        "ai_skipped": bool(row.ai_skipped),
        "raw_features": json.loads(row.raw_features) if row.raw_features else None,
        "feedback": [{"verdict": f.verdict, "notes": f.notes} for f in feedbacks],
    }


@router.get("/", response_model=list[AlertRow])
async def get_alerts(
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    rows = await list_alerts(db, limit=limit, offset=offset)
    return [
        AlertRow(
            id=r.id,
            timestamp=r.timestamp,
            rf_label=r.rf_label,
            rf_conf=r.rf_confidence,
            ae_score=r.ae_score,
            ae_label=r.ae_label,
            consensus=r.consensus,
            severity=r.severity,
        )
        for r in rows
    ]


@router.post("/{alert_id}/feedback")
async def submit_feedback(
    alert_id: int,
    req: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    await insert_feedback(db, alert_id, req.verdict, req.notes)
    return {"status": "ok", "alert_id": alert_id}
