"""
Training trigger API.

POST /training/start  — queue a training job (runs in background thread pool)
GET  /training/status/{run_id}
GET  /training/history
"""
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, AsyncSessionLocal
from app.db import crud
from app.models.schemas import (
    TrainingStartRequest, TrainingRunResponse, TrainingDataSource
)

router = APIRouter(prefix="/training", tags=["training"])

# Path defaults mirror PipelineConfig defaults
_MODELS_DIR   = "models/"
_CICIDS_DIR   = "data/cicids2017/"
_NSLKDD_PATH  = "data/nslkdd/KDDTrain+.txt"


def _serialise_run(row) -> TrainingRunResponse:
    return TrainingRunResponse(
        id=row.id,
        stage=row.stage,
        status=row.status,
        data_source=row.data_source,
        started_at=row.started_at,
        completed_at=row.completed_at,
        metrics=json.loads(row.metrics_json) if row.metrics_json else None,
        error=row.error,
        created_at=row.created_at,
    )


# ── Background training task ──────────────────────────────────────────────────

def _blocking_train(stage: str) -> dict:
    """
    Runs the training pipeline synchronously.  Called via run_in_executor so
    it executes in a thread pool — the event loop stays unblocked.
    """
    from training.pipelines.training_pipeline import TrainingPipeline, PipelineConfig

    config = PipelineConfig(
        cicids_data_dir=_CICIDS_DIR,
        nslkdd_train_path=_NSLKDD_PATH,
        models_dir=_MODELS_DIR,
    )
    pipeline = TrainingPipeline.from_config(config)
    metrics: dict = {}

    if stage in ("rf", "all"):
        results = pipeline.run_rf()
        metrics["rf"] = {
            "macro_f1": round(float(results.macro_f1), 4),
            "accuracy": round(float(results.accuracy), 4),
        }

    if stage in ("ae1", "all"):
        m = pipeline.run_ae_phase1()
        metrics["ae1"] = m or {}

    if stage in ("ae2", "all"):
        m = pipeline.run_ae_phase2()
        metrics["ae2"] = m or {}

    return metrics


async def _run_training_bg(run_id: int, stage: str) -> None:
    """Async wrapper — runs blocking training in thread pool, updates DB."""
    loop = asyncio.get_event_loop()

    async with AsyncSessionLocal() as db:
        await crud.update_training_run(
            db, run_id, status="running",
            started_at=datetime.now(timezone.utc),
        )

    try:
        metrics = await loop.run_in_executor(None, _blocking_train, stage)

        async with AsyncSessionLocal() as db:
            await crud.update_training_run(
                db, run_id,
                status="done",
                completed_at=datetime.now(timezone.utc),
                metrics=metrics,
            )
    except Exception as exc:
        async with AsyncSessionLocal() as db:
            await crud.update_training_run(
                db, run_id,
                status="failed",
                completed_at=datetime.now(timezone.utc),
                error=str(exc)[:2000],
            )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/start", response_model=TrainingRunResponse, status_code=202)
async def start_training(
    req: TrainingStartRequest,
    db: AsyncSession = Depends(get_db),
):
    if req.data_source == TrainingDataSource.db:
        raise HTTPException(
            status_code=501,
            detail="DB data source not yet available. Run scripts/import_datasets.py first, then this will be enabled.",
        )

    run_id = await crud.create_training_run(db, req.stage.value, req.data_source.value)
    asyncio.create_task(_run_training_bg(run_id, req.stage.value))

    row = await crud.get_training_run(db, run_id)
    return _serialise_run(row)


@router.get("/status/{run_id}", response_model=TrainingRunResponse)
async def get_training_status(run_id: int, db: AsyncSession = Depends(get_db)):
    row = await crud.get_training_run(db, run_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Training run {run_id} not found")
    return _serialise_run(row)


@router.get("/history", response_model=list[TrainingRunResponse])
async def get_training_history(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    rows = await crud.list_training_runs(db, limit=limit)
    return [_serialise_run(r) for r in rows]
