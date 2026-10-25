import asyncio
import json
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.preprocessor import preprocessor
from app.services.inference import rf_inference, ae_inference
from app.services.consensus import compute_consensus
from app.config import settings

router = APIRouter(prefix="/stream", tags=["streaming"])


@router.websocket("/ws")
async def stream_predictions(websocket: WebSocket):
    """
    WebSocket endpoint for live/replay flow streaming.
    Client sends: {"features": {...}}
    Server returns: PredictResponse JSON per flow.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            features = payload.get("features", {})

            x = preprocessor.transform(features)
            rf_result = rf_inference.predict(x)
            ae_result = ae_inference.predict(x)
            consensus, severity = compute_consensus(rf_result, ae_result)

            response = {
                "flow_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "rf_label": rf_result.label,
                "rf_confidence": rf_result.confidence,
                "ae_score": ae_result.score,
                "ae_label": ae_result.label,
                "consensus": consensus.value,
                "severity": severity.value,
            }
            await websocket.send_text(json.dumps(response))
            await asyncio.sleep(settings.replay_speed_ms / 1000)
    except WebSocketDisconnect:
        pass
