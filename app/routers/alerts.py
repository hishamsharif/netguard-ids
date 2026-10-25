from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.schemas import AlertRow, FeedbackRequest
from app.db.crud import list_alerts, insert_feedback
import json

router = APIRouter(prefix="/alerts", tags=["alerts"])


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
