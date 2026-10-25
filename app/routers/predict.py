import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.schemas import FlowRequest, PredictResponse, RuleResult, Consensus, Severity
from app.services import rule_engine
from app.services.preprocessor import preprocessor
from app.services.inference import rf_inference, ae_inference
from app.services.consensus import compute_consensus
from app.db.crud import insert_alert, insert_feature_log

router = APIRouter(prefix="/predict", tags=["inference"])


@router.post("/flow", response_model=PredictResponse)
async def predict_flow(req: FlowRequest, db: AsyncSession = Depends(get_db)):
    flow_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc)

    # ── Stage 1: Rule Engine (non-AI, deterministic) ─────────────────
    rule_result_raw = rule_engine.evaluate(req.features)
    rule_result = RuleResult(
        verdict=rule_result_raw.verdict,
        triggered_rules=rule_result_raw.triggered_rules,
        confidence=rule_result_raw.confidence,
        details=rule_result_raw.details,
    )

    if rule_result_raw.verdict == "BENIGN":
        # Fast path — AI pipeline skipped
        response = PredictResponse(
            flow_id=flow_id,
            rule=rule_result,
            rf=None,
            ae=None,
            consensus=Consensus.SAFE,
            severity=Severity.LOW,
            ai_skipped=True,
            timestamp=ts,
        )
        await insert_alert(db, response, req.features)
        return response

    # ── Stage 2: Supervised — Random Forest ──────────────────────────
    try:
        x = preprocessor.transform(req.features)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {e}")

    rf_result = rf_inference.predict(x)

    # ── Stage 3: Unsupervised — LSTM Autoencoder ──────────────────────
    ae_result = ae_inference.predict(x)

    consensus, severity = compute_consensus(rf_result, ae_result)

    response = PredictResponse(
        flow_id=flow_id,
        rule=rule_result,
        rf=rf_result,
        ae=ae_result,
        consensus=consensus,
        severity=severity,
        ai_skipped=False,
        timestamp=ts,
    )
    alert_id = await insert_alert(db, response, req.features)
    await insert_feature_log(db, alert_id, x)
    return response
