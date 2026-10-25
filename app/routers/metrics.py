from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.schemas import MetricsResponse, ThresholdRequest
from app.services.inference import ae_inference
from app.db.crud import get_metrics_counts, log_threshold_change

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/", response_model=MetricsResponse)
async def get_metrics(db: AsyncSession = Depends(get_db)):
    counts = await get_metrics_counts(db)
    tp = counts["tp"] or 0
    fp = counts["fp"] or 0
    fn = counts["fn"] or 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None
    fpr = fp / (fp + (counts["total"] - tp - fn)) if counts["total"] > 0 else None

    return MetricsResponse(
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        total_alerts=counts["total"],
        feedback_count=counts["feedback_count"],
        ae_threshold=ae_inference.threshold,
    )


@router.put("/threshold")
async def update_threshold(req: ThresholdRequest, db: AsyncSession = Depends(get_db)):
    old = ae_inference.threshold
    await log_threshold_change(db, old, req.threshold)
    ae_inference.threshold = req.threshold
    return {"old_threshold": old, "new_threshold": req.threshold}
