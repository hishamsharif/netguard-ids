import uuid
import csv
import io
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.schemas import FlowRequest, PredictResponse, RuleResult, Consensus, Severity
from app.services import rule_engine
from app.services.preprocessor import preprocessor
from app.services.inference import rf_inference, ae_inference
from app.services.consensus import compute_consensus
from app.db.crud import insert_alert, insert_feature_log

router = APIRouter(prefix="/predict", tags=["inference"])


async def _infer_single(features: dict, db: AsyncSession) -> PredictResponse:
    """Core 3-stage pipeline; shared by /flow and /batch."""
    flow_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc)

    rule_result_raw = rule_engine.evaluate(features)
    rule_result = RuleResult(
        verdict=rule_result_raw.verdict,
        triggered_rules=rule_result_raw.triggered_rules,
        confidence=rule_result_raw.confidence,
        details=rule_result_raw.details,
    )

    if rule_result_raw.verdict == "BENIGN":
        response = PredictResponse(
            flow_id=flow_id, rule=rule_result, rf=None, ae=None,
            consensus=Consensus.SAFE, severity=Severity.LOW,
            ai_skipped=True, timestamp=ts,
        )
        await insert_alert(db, response, features)
        return response

    try:
        x = preprocessor.transform(features)
    except Exception as e:
        raise ValueError(f"Preprocessing: {e}")

    rf_result = rf_inference.predict(x)
    ae_result = ae_inference.predict(x)
    consensus, severity = compute_consensus(rf_result, ae_result)

    response = PredictResponse(
        flow_id=flow_id, rule=rule_result, rf=rf_result, ae=ae_result,
        consensus=consensus, severity=severity, ai_skipped=False, timestamp=ts,
    )
    alert_id = await insert_alert(db, response, features)
    await insert_feature_log(db, alert_id, x)
    return response


@router.post("/flow", response_model=PredictResponse)
async def predict_flow(req: FlowRequest, db: AsyncSession = Depends(get_db)):
    try:
        return await _infer_single(req.features, db)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/batch")
async def predict_batch(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """
    Accept a CSV file where column headers are CIC-IDS2017 feature names.
    Returns per-row inference results plus a consensus breakdown summary.
    """
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")  # handle Excel BOM
        rows = list(csv.DictReader(io.StringIO(text)))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"CSV parse error: {e}")

    if not rows:
        raise HTTPException(status_code=422, detail="CSV file is empty or has no data rows")

    results, errors = [], []
    for i, row in enumerate(rows):
        try:
            features = {
                k.strip(): float(v.strip())
                for k, v in row.items()
                if k and v and v.strip() not in ("", "nan", "NaN", "inf")
            }
            res = await _infer_single(features, db)
            results.append({
                "row": i + 1,
                "flow_id": res.flow_id,
                "consensus": res.consensus.value,
                "severity": res.severity.value,
                "rf_label": res.rf.label if res.rf else "SKIPPED",
                "rf_conf": round(res.rf.confidence, 4) if res.rf else 0.0,
                "ae_score": round(res.ae.score, 6) if res.ae else 0.0,
                "ai_skipped": res.ai_skipped,
                "rules": res.rule.triggered_rules,
            })
        except Exception as e:
            errors.append({"row": i + 1, "error": str(e)})

    consensus_counts: dict[str, int] = {}
    for r in results:
        k = r["consensus"]
        consensus_counts[k] = consensus_counts.get(k, 0) + 1

    return {
        "total": len(rows),
        "processed": len(results),
        "error_count": len(errors),
        "consensus_counts": consensus_counts,
        "results": results,
        "errors": errors,
    }
