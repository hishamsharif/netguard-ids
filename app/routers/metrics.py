import json
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.schemas import MetricsResponse, ThresholdRequest
from app.services.inference import ae_inference
from app.db.crud import get_metrics_counts, log_threshold_change, get_active_model_version

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/", response_model=MetricsResponse)
async def get_metrics(db: AsyncSession = Depends(get_db)):
    counts = await get_metrics_counts(db)
    tp = counts["tp"] or 0
    fp = counts["fp"] or 0
    fn = counts["fn"] or 0

    # Live metrics from analyst feedback (only when feedback exists)
    live_precision = tp / (tp + fp) if (tp + fp) > 0 else None
    live_recall    = tp / (tp + fn) if (tp + fn) > 0 else None
    live_f1        = (2 * live_precision * live_recall / (live_precision + live_recall)) if (live_precision and live_recall) else None
    fpr = fp / (fp + (counts["total"] - tp - fn)) if counts["total"] > 0 else None

    # Baseline metrics from the active model version (training eval report)
    baseline_precision = baseline_recall = baseline_f1 = None
    active = await get_active_model_version(db)
    if active and active.metrics_json:
        bm = json.loads(active.metrics_json)
        m = bm.get("metrics", {})
        baseline_precision = m.get("precision_macro")
        baseline_recall    = m.get("recall_macro")
        baseline_f1        = m.get("f1_macro")

    return MetricsResponse(
        precision=live_precision if counts["feedback_count"] > 0 else baseline_precision,
        recall=live_recall if counts["feedback_count"] > 0 else baseline_recall,
        f1=live_f1 if counts["feedback_count"] > 0 else baseline_f1,
        false_positive_rate=fpr,
        total_alerts=counts["total"],
        feedback_count=counts["feedback_count"],
        ae_threshold=ae_inference.threshold,
        baseline_precision=baseline_precision,
        baseline_recall=baseline_recall,
        baseline_f1=baseline_f1,
    )


@router.put("/threshold")
async def update_threshold(req: ThresholdRequest, db: AsyncSession = Depends(get_db)):
    old = ae_inference.threshold
    await log_threshold_change(db, old, req.threshold)
    ae_inference.threshold = req.threshold
    return {"old_threshold": old, "new_threshold": req.threshold}
