from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from app.database import init_db
from app.services.preprocessor import preprocessor
from app.services.inference import rf_inference, ae_inference
from app.services.explainer import explainer
from app.routers import predict, explain, alerts, stream, metrics
from app.routers import training
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
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

app.mount("/static", StaticFiles(directory="ui/static"), name="static")
templates = Jinja2Templates(directory="ui/templates")


@app.get("/", include_in_schema=False)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "version": app.version}
