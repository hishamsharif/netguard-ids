"""
Model version registry — list, inspect and activate trained snapshots.
RF and AE versions can be activated independently.
"""
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database import get_db
from app.db.models import AlertLog, ModelVersion
from app.db.crud import (
    list_model_versions, activate_model_version,
    activate_rf_version, activate_ae_version,
    get_active_model_version, count_pending_logs,
)

router = APIRouter(prefix="/models", tags=["models"])

_RELOAD_NOTE = "Version activated in DB. Model files reload on next service restart."


def _mv_to_dict(v: ModelVersion) -> dict:
    return {
        "id": v.id,
        "version_tag": v.version_tag,
        "stage": v.stage,
        "is_active": bool(v.is_active),
        "is_active_rf": bool(getattr(v, "is_active_rf", 0)),
        "is_active_ae": bool(getattr(v, "is_active_ae", 0)),
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
    active_rf = next((v for v in versions if getattr(v, "is_active_rf", 0)), None)
    active_ae = next((v for v in versions if getattr(v, "is_active_ae", 0)), None)
    pending = await count_pending_logs(db, active.trained_up_to_alert_id if active else 0)
    total_alerts = await db.scalar(
        select(func.count()).select_from(AlertLog).where(AlertLog.ai_skipped == 0)
    )
    return {
        "versions": [_mv_to_dict(v) for v in versions],
        "pending_logs": pending,
        "total_inference_logs": total_alerts or 0,
        "active_rf_id": active_rf.id if active_rf else None,
        "active_ae_id": active_ae.id if active_ae else None,
    }


@router.post("/{version_id}/activate")
async def activate_version(version_id: int, db: AsyncSession = Depends(get_db)):
    """Activate a version for both RF and AE (unified switch)."""
    version = await db.get(ModelVersion, version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Model version not found")
    updated = await activate_model_version(db, version_id)
    return {"status": "ok", "active": _mv_to_dict(updated), "note": _RELOAD_NOTE}


@router.post("/{version_id}/activate/rf")
async def activate_rf(version_id: int, db: AsyncSession = Depends(get_db)):
    """Activate a version's RF model independently (AE version unchanged)."""
    version = await db.get(ModelVersion, version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Model version not found")
    if not version.model_path_rf:
        raise HTTPException(status_code=422, detail="Version has no RF model path")
    updated = await activate_rf_version(db, version_id)
    return {"status": "ok", "active_rf": _mv_to_dict(updated), "note": _RELOAD_NOTE}


@router.post("/{version_id}/activate/ae")
async def activate_ae(version_id: int, db: AsyncSession = Depends(get_db)):
    """Activate a version's AE model independently (RF version unchanged)."""
    version = await db.get(ModelVersion, version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Model version not found")
    if not version.model_path_ae:
        raise HTTPException(status_code=422, detail="Version has no AE model path")
    updated = await activate_ae_version(db, version_id)
    return {"status": "ok", "active_ae": _mv_to_dict(updated), "note": _RELOAD_NOTE}
