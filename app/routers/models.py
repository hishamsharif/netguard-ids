"""
Model version registry — list, inspect and activate trained snapshots.
"""
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database import get_db
from app.db.models import AlertLog, ModelVersion
from app.db.crud import (
    list_model_versions, activate_model_version,
    get_active_model_version, count_pending_logs,
)

router = APIRouter(prefix="/models", tags=["models"])


def _mv_to_dict(v: ModelVersion) -> dict:
    return {
        "id": v.id,
        "version_tag": v.version_tag,
        "stage": v.stage,
        "is_active": bool(v.is_active),
        "trained_on_source": v.trained_on_source,
        "trained_flow_count": v.trained_flow_count,
        "model_path_rf": v.model_path_rf,
        "model_path_ae": v.model_path_ae,
        "metrics": json.loads(v.metrics_json) if v.metrics_json else None,
        "training_run_id": v.training_run_id,
        "notes": v.notes,
        "created_at": v.created_at,
    }


@router.get("/")
async def list_versions(db: AsyncSession = Depends(get_db)):
    versions = await list_model_versions(db)
    active = next((v for v in versions if v.is_active), None)
    pending = await count_pending_logs(db, active.trained_up_to_alert_id if active else 0)
    total_alerts = await db.scalar(
        select(func.count()).select_from(AlertLog).where(AlertLog.ai_skipped == 0)
    )
    return {
        "versions": [_mv_to_dict(v) for v in versions],
        "pending_logs": pending,
        "total_inference_logs": total_alerts or 0,
    }


@router.post("/{version_id}/activate")
async def activate_version(version_id: int, db: AsyncSession = Depends(get_db)):
    """
    Switch the active model version. The server must be restarted (or reload
    triggered) for inference services to pick up the new model files.
    In practice: update is_active flag; the next restart loads the new paths.
    """
    version = await db.get(ModelVersion, version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Model version not found")
    updated = await activate_model_version(db, version_id)
    return {
        "status": "ok",
        "active": _mv_to_dict(updated),
        "note": "Restart the API service to load the new model files into memory.",
    }
