from fastapi import APIRouter, HTTPException
from app.models.schemas import FlowRequest, ExplainResponse
from app.services.preprocessor import preprocessor
from app.services.inference import rf_inference, ae_inference
from app.services.consensus import compute_consensus
from app.services.explainer import explainer
import uuid

router = APIRouter(prefix="/explain", tags=["explainability"])


@router.post("/flow", response_model=ExplainResponse)
async def explain_flow(req: FlowRequest):
    try:
        x = preprocessor.transform(req.features)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    rf_result = rf_inference.predict(x)
    ae_result = ae_inference.predict(x)
    consensus, _ = compute_consensus(rf_result, ae_result)

    rf_imp = explainer.rf_importance()
    ae_errors = explainer.ae_feature_error(x)
    nl = explainer.natural_language(rf_imp, ae_errors, consensus.value)

    return ExplainResponse(
        flow_id=str(uuid.uuid4()),
        rf_importance=rf_imp,
        ae_recon_error=ae_errors,
        natural_language=nl,
    )
