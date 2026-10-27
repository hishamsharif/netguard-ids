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


async def _infer_single(features: dict, db: AsyncSession) -> tuple[PredictResponse, int]:
    """Core 3-stage pipeline; returns (response, alert_id)."""
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
        alert_id = await insert_alert(db, response, features)
        return response, alert_id

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
    return response, alert_id


@router.post("/flow", response_model=PredictResponse)
async def predict_flow(req: FlowRequest, db: AsyncSession = Depends(get_db)):
    try:
        response, _ = await _infer_single(req.features, db)
        return response
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

    # Validate CSV columns against required vs optional feature split.
    # Required = RF Gini importance > 1% (42 features). Optional = remaining 28.
    # Missing required → 422. Missing optional → silently substitute training median.
    all_features = preprocessor.feature_names
    if all_features:
        csv_cols = {k.strip() for k in rows[0].keys()}
        required_features = [
            f for f in all_features
            if f not in {
                "Bwd IAT Mean", "Flow Packets/s", "Fwd Header Length", "Avg Bwd Segment Size",
                "Bwd IAT Total", "Bwd IAT Min", "Idle Std", "Bwd IAT Std", "Active Max",
                "Fwd Packets/s", "SYN Flag Count", "Idle Mean", "act_data_pkt_fwd",
                "Bwd IAT Max", "RST Flag Count", "Fwd Packet Length Min", "PSH Flag Count",
                "CWE Flag Count", "Active Mean", "Fwd PSH Flags", "Packet Length Variance",
                "min_seg_size_forward", "Active Min", "Packet Length Std", "ACK Flag Count",
                "Bwd PSH Flags", "FIN Flag Count", "URG Flag Count",
                "Idle Max", "Idle Min",
            }
        ]
        missing_required = [f for f in required_features if f not in csv_cols]
        if missing_required:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"CSV is missing {len(missing_required)} required feature column(s). "
                               f"Optional features (low importance) may be omitted — training median is substituted.",
                    "missing_required_columns": missing_required,
                    "hint": "Column names must match CIC-IDS2017 feature names exactly (case-sensitive). "
                            "Use the format hint in the UI for the full 72-column template.",
                },
            )

    results, errors = [], []
    for i, row in enumerate(rows):
        try:
            features = {
                k.strip(): float(v.strip())
                for k, v in row.items()
                if k and v and v.strip() not in ("", "nan", "NaN", "inf")
            }
            res, alert_id = await _infer_single(features, db)
            results.append({
                "row": i + 1,
                "alert_id": alert_id,
                "flow_id": res.flow_id,
                "consensus": res.consensus.value,
                "severity": res.severity.value,
                "rule_verdict": res.rule.verdict.value,
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
