import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
from app.main import app
from app.database import engine, Base
from app.models.schemas import RFResult, AEResult


SAMPLE_FEATURES = {f"feature_{i}": float(i) for i in range(79)}


@pytest_asyncio.fixture
async def client():
    """Create tables in a fresh in-memory-ish DB, then yield an ASGI test client."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    with (
        patch("app.services.preprocessor.preprocessor.load"),
        patch("app.services.inference.rf_inference.load"),
        patch("app.services.inference.ae_inference.load"),
        patch("app.services.explainer.explainer.load"),
        patch("app.database.init_db", new_callable=AsyncMock),   # skip lifespan re-init
        patch("app.services.preprocessor.preprocessor._loaded", True),
        patch("app.services.preprocessor.preprocessor.transform",
              return_value=[[0.0] * 79]),
        patch("app.services.inference.rf_inference.predict",
              return_value=RFResult(label="BENIGN", confidence=0.95, top_features=[])),
        patch("app.services.inference.ae_inference.predict",
              return_value=AEResult(score=0.01, threshold=0.023, label="BENIGN")),
        patch("app.services.rule_engine.evaluate",
              return_value=type("R", (), {
                  "verdict": "SUSPICIOUS", "triggered_rules": ["R-06:high_risk_port"],
                  "confidence": 0.25, "details": {}
              })()),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac


@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_predict_flow_safe(client):
    r = await client.post("/predict/flow", json={"features": SAMPLE_FEATURES})
    assert r.status_code == 200
    body = r.json()
    assert body["consensus"] == "SAFE"


@pytest.mark.asyncio
async def test_alerts_list(client):
    r = await client.get("/alerts/")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


@pytest.mark.asyncio
async def test_metrics(client):
    r = await client.get("/metrics/")
    assert r.status_code == 200
    assert "ae_threshold" in r.json()
