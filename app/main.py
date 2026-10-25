import json
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from sqlalchemy import text
from app.database import init_db, AsyncSessionLocal
from app.services.preprocessor import preprocessor
from app.services.inference import rf_inference, ae_inference
from app.services.explainer import explainer
from app.routers import predict, explain, alerts, stream, metrics
from app.routers import training
from app.routers import models as models_router
from app.config import settings


async def _seed_baseline() -> None:
    """
    On startup: if no ModelVersion exists yet AND trained model files are present,
    seed the v1.0-baseline registry entry.  If model files are missing the DB
    entry is still seeded with notes indicating training is required; the UI
    surfaces this as a 'Train Baseline' prompt instead of the inference tabs.
    """
    from app.db.crud import seed_baseline_model_version
    rf_ready  = Path(settings.rf_model_path).exists()
    ae_ready  = Path(settings.ae_model_path).exists()
    eval_path = Path(settings.rf_model_path).parent / "reports" / "eval_rf.json"
    baseline_metrics: dict = {}
    if eval_path.exists():
        with open(eval_path) as f:
            baseline_metrics = json.load(f)
    notes = (
        "Initial baseline trained on CIC-IDS2017 (RF) + NSL-KDD→CIC-IDS2017 transfer (LSTM-AE)"
        if (rf_ready and ae_ready) else
        "Model files not found — use the Training tab to run baseline training first"
    )
    async with AsyncSessionLocal() as db:
        await seed_baseline_model_version(db, baseline_metrics, notes=notes, models_ready=(rf_ready and ae_ready))


async def _migrate_db() -> None:
    """Add columns introduced after initial schema — idempotent (silently skips existing)."""
    from app.database import engine
    new_cols = [
        "ALTER TABLE model_version ADD COLUMN is_active_rf INTEGER DEFAULT 0",
        "ALTER TABLE model_version ADD COLUMN is_active_ae INTEGER DEFAULT 0",
    ]
    async with engine.begin() as conn:
        for sql in new_cols:
            try:
                await conn.execute(text(sql))
            except Exception:
                pass  # column already exists


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await _migrate_db()
    await _seed_baseline()
    preprocessor.load()
    rf_inference.load()
    ae_inference.load()
    explainer.load()
    yield


app = FastAPI(
    title="NetGuard IDS",
    description="Network Intrusion Detection — RF + LSTM-AE dual-model SOC tool",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(alerts.router)
app.include_router(stream.router)
app.include_router(metrics.router)
app.include_router(training.router)
app.include_router(models_router.router)

app.mount("/static", StaticFiles(directory="ui/static"), name="static")
templates = Jinja2Templates(directory="ui/templates")


@app.get("/", include_in_schema=False)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "version": app.version}
