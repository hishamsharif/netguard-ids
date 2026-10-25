# NetGuard IDS

A network intrusion detection system (NIDS) built for **CS7003NU Assessment 2**.  
Dual-model SOC pipeline: **Random Forest** (known-pattern detection) +
**LSTM Autoencoder** (anomaly / zero-day detection), with a FastAPI backend,
live WebSocket streaming, explainability, an analyst dashboard, and an MCP
server for AI-assistant integration.

- **Design decisions & future extension queue** → [`DECISIONS.md`](DECISIONS.md)
- **Interactive API docs** (when running) → http://localhost:8000/docs

---

## Contents

1. [System Overview](#1-system-overview)
2. [Prerequisites](#2-prerequisites)
3. [First-time Setup](#3-first-time-setup)
4. [Environment Configuration](#4-environment-configuration)
5. [Running the Application](#5-running-the-application)
6. [Daily Operations — ops.sh](#6-daily-operations--opssh)
7. [Container Operations & Debugging](#7-container-operations--debugging)
8. [Training Pipeline](#8-training-pipeline)
9. [Running Tests](#9-running-tests)
10. [API Reference](#10-api-reference)
11. [Project Structure](#11-project-structure)
12. [Feedback-Label Bridge](#12-feedback-label-bridge)

---

## 1. System Overview

### Detection pipeline

```
Network flow (CIC-IDS2017 features)
           │
   ┌───────▼────────┐
   │  Rule Engine   │  deterministic bypass — benign flows exit here (fast path)
   └───────┬────────┘
           │ SUSPICIOUS
   ┌───────▼────────┐    ┌────────────────────────┐
   │ Random Forest  │    │  LSTM Autoencoder      │
   │ (supervised)   │    │  (unsupervised anomaly) │
   └───────┬────────┘    └──────────┬─────────────┘
           └──────────┬─────────────┘
                ┌─────▼──────────┐
                │ Consensus      │  SAFE · CONFIRMED · KNOWN_PATTERN · ZERO_DAY
                └─────┬──────────┘
                      │
                ┌─────▼──────────┐
                │  FastAPI REST  │  + WebSocket streaming
                └─────┬──────────┘
           ┌──────────┴──────────┐
     Dashboard UI           MCP Server
     (Nginx + Alpine.js)    (FastMCP / Claude)
```

### Transfer learning

```
NSL-KDD (39 features)          CIC-IDS2017 benign (70 features, 2.27 M rows)
       │                                   │
  Phase 1: pretrain          Phase 2: fine-tune (encoder frozen ep 1-9,
  LSTM-AE on normal     ──►  partial weight transfer, clip inputs to [-10,10])
  traffic                    threshold = p95 normal reconstruction error
```

### Database

| Mode | Database | Setup required |
|------|----------|----------------|
| Local dev (`ops.sh start`) | **SQLite** — `netguard_dev.db` file | None — created automatically |
| Docker (`ops.sh start --docker`) | **PostgreSQL 16** — `netguard_pg` volume | None — created automatically |

> **PostgreSQL for local dev** is a future extension (FE-002, [`DECISIONS.md`](DECISIONS.md)).
> It becomes relevant once dataset tables are added to support DB-sourced training.
> Until then, SQLite covers all operational tables (alerts, feedback, feature vectors, training runs).

---

## 2. Prerequisites

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB (training uses up to 12 GB) |
| GPU | None (CPU inference works) | NVIDIA CUDA — needed for practical training |
| Disk | 5 GB (models only) | 20 GB (models + datasets) |

### Software

| Tool | Version | Required for |
|------|---------|--------------|
| Miniconda / Anaconda | Any recent | Python env (`dl_env`) |
| Python | 3.12 (via conda) | API, training, tests |
| Docker Desktop / Engine | ≥ 24 with Compose v2 | Docker mode |
| CUDA Toolkit | ≥ 11.8 | GPU training |
| WSL2 (Windows only) | Any | Running Linux tooling on Windows |

---

## 3. First-time Setup

### Step 1 — Clone / enter the project

```bash
cd dev/netguard-ids
```

### Step 2 — Run setup script

```bash
# Standard setup (conda env + Python deps + Docker images + .env + dirs)
bash setup.sh

# Options:
bash setup.sh --docker-only   # skip conda/pip, just build Docker images
bash setup.sh --local-only    # skip Docker build
```

> **PostgreSQL local setup** (`--with-pg`) is a future extension (FE-002).
> It becomes relevant once DB-sourced training datasets are supported.
> The default stack uses SQLite locally and PostgreSQL automatically in Docker.

What `setup.sh` does:

1. Verifies / installs Miniconda
2. Creates `dl_env` conda environment (Python 3.12)
3. Installs all Python dependencies from `requirements.txt`
4. Builds Docker images (`docker compose build`)
5. Creates `.env` from template if not present
6. Creates runtime directories (`models/`, `data/`, `logs/`)

### Step 3 — Place trained models

Training from scratch takes ~45 minutes with a GPU. If you have pre-trained
artifacts, copy them into `models/`:

```
models/
  rf_cicids.joblib          Random Forest classifier
  lstm_ae_cicids.pt         LSTM-AE (fine-tuned, Phase 2)
  lstm_ae_nslkdd.pt         LSTM-AE (pretrained, Phase 1)
  scaler.joblib             RobustScaler fitted on CIC-IDS2017
  label_encoder.joblib      RF label encoder
  feature_names.json        Ordered feature name list (70 names)
```

If models are missing, the API starts but returns 500 on `/predict/flow`.
Train them first — see [§8 Training Pipeline](#8-training-pipeline).

### Step 4 — Verify

```bash
conda activate dl_env
PYTHONPATH=. python -m pytest tests/ -v
# Expected: 20 passed
```

---

## 4. Environment Configuration

All runtime settings live in `.env` at the project root.
`docker-compose.yml` overrides `DATABASE_URL` automatically for Docker mode —
you do not need to change `.env` when switching between local and Docker.

### `.env` reference

| Variable | Local default | Docker override | Description |
|----------|--------------|-----------------|-------------|
| `APP_ENV` | `development` | — | `development` or `production` |
| `APP_HOST` | `0.0.0.0` | — | Bind address for uvicorn |
| `APP_PORT` | `8000` | — | API listen port |
| `DATABASE_URL` | `sqlite+aiosqlite:///./netguard_dev.db` | `postgresql+asyncpg://netguard:netguard_dev@db:5432/netguard` | SQLAlchemy async DB URL |
| `RF_MODEL_PATH` | `models/rf_cicids.joblib` | — | Path to trained RF model |
| `AE_MODEL_PATH` | `models/lstm_ae_cicids.pt` | — | Path to trained LSTM-AE model |
| `SCALER_PATH` | `models/scaler.joblib` | — | Path to fitted RobustScaler |
| `AE_THRESHOLD` | `0.008927` | — | Anomaly threshold (p95 of normal reconstruction error from Phase 2 training) |
| `REPLAY_SPEED_MS` | `200` | — | Delay between WebSocket replay frames (ms) |
| `MCP_PORT` | `8001` | — | FastMCP server port |

### Switching databases manually

```bash
# Local dev — SQLite (default, no setup needed)
DATABASE_URL=sqlite+aiosqlite:///./netguard_dev.db

# Docker internal (set automatically by docker-compose.yml — no manual change needed)
DATABASE_URL=postgresql+asyncpg://netguard:netguard_dev@db:5432/netguard
```

> **Local PostgreSQL** is a future extension (FE-002, [`DECISIONS.md`](DECISIONS.md)).
> Docker mode handles PostgreSQL automatically — no `.env` edit required.

> **Note:** Tables are created automatically on first startup via `init_db()`.
> You never need to run migrations manually.

---

## 5. Running the Application

### 5.1 Local mode (SQLite, no Docker)

```bash
bash ops.sh start
```

Starts: API on `:8000`, MCP on `:8001`. Database is SQLite (`netguard_dev.db`) — no service needed.

| Service | URL / endpoint |
|---------|---------------|
| Analyst dashboard | http://localhost:8000 |
| OpenAPI docs | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |
| MCP server | localhost:8001 (SSE transport) |

### 5.2 Docker mode (PostgreSQL, all 4 containers)

```bash
bash ops.sh start --docker
```

Starts four containers in dependency order: `db` → `api` → `ui` + `mcp`.

| Container | Image | Port | Role |
|-----------|-------|------|------|
| `netguard-db` | postgres:16-alpine | 5432 | PostgreSQL — persistent volume `netguard_pg` |
| `netguard-api` | netguard-ids (Dockerfile) | 8000 | FastAPI inference + training backend |
| `netguard-ui` | netguard-ids-ui (Dockerfile.ui) | 80 | Nginx — static assets + reverse proxy to api |
| `netguard-mcp` | netguard-ids (Dockerfile) | 8001 | FastMCP server |

| URL | What |
|-----|------|
| http://localhost | Analyst dashboard (via Nginx) |
| http://localhost:8000/docs | OpenAPI interactive docs |
| http://localhost:8000/health | Health check |
| localhost:8001 | MCP server |

> **Models and data are volume-mounted read-only** (`./models:/app/models:ro`,
> `./data:/app/data:ro`). Build the models on the host before starting Docker.

---

## 6. Daily Operations — ops.sh

All day-to-day commands go through `ops.sh`. Add `--docker` to any command to
route through Docker Compose instead of local processes.

### Start / stop / restart

```bash
# Start everything
bash ops.sh start                  # local
bash ops.sh start --docker         # Docker

# Stop everything
bash ops.sh stop                   # local (DB left running by default)
bash ops.sh stop --docker          # Docker (removes containers, keeps volumes)

# Restart all or one service
bash ops.sh restart                # local — restarts api + mcp
bash ops.sh restart api            # local — restart api only
bash ops.sh restart --docker       # Docker — restart all
bash ops.sh restart --docker api   # Docker — restart api container only
```

### Status

```bash
bash ops.sh status                 # local — shows PID and health for each service
bash ops.sh status --docker        # Docker — equivalent to docker compose ps
```

Example local output:

```
  Service      Status           PID     Endpoint
  ─────────────────────────────────────────────────────
  api          running          18432   http://localhost:8000
               health OK
  mcp          running          18433   localhost:8001
```

> Local mode uses SQLite — there is no `db` service to start or check.

### Logs

```bash
bash ops.sh logs                   # tail api log (local)
bash ops.sh logs mcp               # tail mcp log (local)
bash ops.sh logs --docker          # tail all container logs
bash ops.sh logs --docker api      # tail api container log only
```

Local log files: `logs/api.log`, `logs/mcp.log`

### Database (Docker / future PostgreSQL only)

> Local mode uses SQLite — no database service to manage.
> The `db` subcommands apply when using Docker mode or, in future, local PostgreSQL (FE-002).

```bash
bash ops.sh db start               # start local PostgreSQL service (future: --with-pg)
bash ops.sh db stop                # stop local PostgreSQL service
bash ops.sh db status              # check if PostgreSQL is accepting connections
```

### Trigger training

```bash
bash ops.sh train                  # all stages (rf + ae1 + ae2)
bash ops.sh train rf               # Random Forest only
bash ops.sh train ae1              # LSTM-AE Phase 1 only (NSL-KDD pretrain)
bash ops.sh train ae2              # LSTM-AE Phase 2 only (CIC-IDS2017 fine-tune)
```

If the API is running, training is triggered via `POST /training/start` and
tracked in the DB. If the API is not running, the CLI pipeline runs directly.

---

## 7. Container Operations & Debugging

### Check container health

```bash
docker compose ps                          # status + health of all containers
docker inspect --format='{{.State.Health.Status}}' netguard-api
```

### Open a shell inside a container

```bash
# API / MCP container (Python app)
docker exec -it netguard-api bash

# PostgreSQL container
docker exec -it netguard-db bash

# Nginx UI container
docker exec -it netguard-ui sh             # Alpine uses sh, not bash
```

### Run Python one-liners inside the API container

```bash
# Check which models are loaded
docker exec netguard-api python -c "
from app.services.inference import rf_inference, ae_inference
rf_inference.load(); ae_inference.load()
print('RF:', rf_inference.model)
print('AE n_features:', ae_inference.model.n_features)
"

# Check DB connectivity
docker exec netguard-api python -c "
import asyncio
from app.database import engine
async def t():
    async with engine.connect() as c:
        print('DB OK — driver:', engine.dialect.name)
asyncio.run(t())
"
```

### Access the database

```bash
# Connect to PostgreSQL inside the db container
docker exec -it netguard-db psql -U netguard -d netguard

# Useful queries once connected
\dt                                        -- list tables
SELECT count(*) FROM alert_log;
SELECT id, stage, status, metrics_json FROM training_run ORDER BY id DESC LIMIT 5;
SELECT id, alert_id FROM feature_log LIMIT 3;
\q                                         -- exit
```

### View and follow logs

```bash
docker compose logs -f                     # all services
docker compose logs -f api                 # api only
docker compose logs -f --tail=50 api       # last 50 lines then follow
```

### Rebuild after code changes

```bash
# Rebuild a single service (e.g. after editing app/ code)
docker compose build api
docker compose up -d api                   # replace running container

# Full rebuild
docker compose down
docker compose up --build -d
```

### Common issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `api` container exits immediately | Model files missing from `models/` | Train models first; check `docker compose logs api` |
| `api` health check failing | DB not ready yet | Wait — `api` waits for `db` healthy (up to 50 s) |
| `ui` returns 502 Bad Gateway | `api` not healthy | Check `docker compose ps`; inspect `api` logs |
| `FATAL: password authentication failed` | Wrong DB URL | Check `DATABASE_URL` env in the container: `docker exec netguard-api env \| grep DATABASE` |
| Port 80 already in use | Another web server running | Stop it or change the `ui` port mapping in `docker-compose.yml` |
| Training OOM kill (exit 144) | Previous training process still in memory | Wait for old process to exit; check `htop` |

### Stop and clean up

```bash
# Stop containers, keep volumes (data survives)
docker compose down

# Stop containers and delete volumes (wipes DB data)
docker compose down -v

# Remove built images
docker compose down --rmi all
```

---

## 8. Training Pipeline

### Datasets required

| Dataset | Path | Download |
|---------|------|----------|
| CIC-IDS2017 | `data/cicids2017/` (8 CSV files, ~844 MB) | [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) |
| NSL-KDD | `data/nslkdd/KDDTrain+.txt`, `KDDTest+.txt` | [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) |

### Train via ops.sh (recommended)

```bash
# API must be running; monitors via the Training tab at http://localhost:8000
bash ops.sh train all              # all 3 stages sequentially
bash ops.sh train rf               # Random Forest only (~4 min)
bash ops.sh train ae1              # LSTM-AE Phase 1 (~1 min, GPU)
bash ops.sh train ae2              # LSTM-AE Phase 2 (~43 min, GPU)
```

### Train via CLI (reliable for long runs)

```bash
conda activate dl_env

# Stage 1 — Random Forest
PYTHONPATH=. python -m training.train_rf \
    --data data/cicids2017/ --out models/rf_cicids.joblib

# Stage 2 — LSTM-AE Phase 1: pretrain on NSL-KDD
PYTHONPATH=. python -m training.pretrain_nslkdd \
    --data data/nslkdd/KDDTrain+.txt --out models/lstm_ae_nslkdd.pt

# Stage 3 — LSTM-AE Phase 2: fine-tune on CIC-IDS2017
PYTHONPATH=. python -m training.finetune_cicids \
    --data data/cicids2017/ --models models/ \
    --epochs 20 --batch 512 --unfreeze-epoch 10
```

After Phase 2, update `AE_THRESHOLD` in `.env` with the printed value.

### Train via UI

1. Open http://localhost:8000 → **Training** tab
2. Select stage and data source (`CSV`)
3. Click **▶ Start Training** — status polls every 3 seconds

> Training jobs run in a background thread. If the API restarts mid-run the
> job does not resume. For runs > 10 min use the CLI or `ops.sh train`.

### Results (trained 2026-10-24)

| Stage | Dataset | Result | Threshold | Time |
|-------|---------|--------|-----------|------|
| RF | CIC-IDS2017 | Macro F1 = 0.887, Accuracy = 99.86% | — | 217.9 s |
| AE Phase 1 | NSL-KDD normal | val_loss converging | 1.9415 | 47 s (GPU) |
| AE Phase 2 | CIC-IDS2017 benign (2.27 M rows) | val_loss = 0.0021 | **0.008927** | 2611.4 s (GPU) |

---

## 9. Running Tests

Tests use SQLite (in-memory via `conftest.py`) — no database service required.

```bash
conda activate dl_env
PYTHONPATH=. python -m pytest tests/ -v
```

Expected output: **20 passed**.

```
tests/test_api.py::test_health                                    PASSED
tests/test_api.py::test_predict_flow_safe                         PASSED
tests/test_api.py::test_alerts_list                               PASSED
tests/test_api.py::test_metrics                                   PASSED
tests/test_inference.py::test_lstm_ae_forward_shape               PASSED
tests/test_inference.py::test_lstm_ae_reconstruction_error_shape  PASSED
tests/test_inference.py::test_lstm_ae_freeze_unfreeze             PASSED
tests/test_inference.py::test_consensus_*  (4 tests)             PASSED
tests/test_inference.py::test_rule_engine_*  (7 tests)           PASSED
tests/test_mcp.py::test_*  (3 tests)                             PASSED
```

---

## 10. API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/predict/flow` | Classify a single flow: rule → RF → AE → consensus |
| `POST` | `/explain/flow` | Feature importance (RF global) + AE per-feature reconstruction error |
| `GET` | `/alerts/` | Alert log — params: `limit`, `offset` |
| `POST` | `/alerts/{id}/feedback` | Submit analyst verdict: `TRUE_POSITIVE` / `FALSE_POSITIVE` / `FALSE_NEGATIVE` |
| `GET` | `/metrics/` | Live precision, recall, F1, FPR, alert counts, AE threshold |
| `PUT` | `/metrics/threshold` | Update AE anomaly threshold at runtime |
| `WS` | `/stream/ws` | WebSocket — send `{"features":{…}}`, receive predictions |
| `POST` | `/training/start` | Queue a training job — body: `{"stage":"rf|ae1|ae2|all","data_source":"csv"}` |
| `GET` | `/training/status/{id}` | Poll training run by ID |
| `GET` | `/training/history` | Last N training runs — param: `limit` |

Full interactive docs with request/response schemas: **http://localhost:8000/docs**

### Example: predict a flow

```bash
curl -s -X POST http://localhost:8000/predict/flow \
  -H "Content-Type: application/json" \
  -d '{"features": {"Destination Port": 22, "Flow Duration": 100, "SYN Flag Count": 5}}' \
  | python3 -m json.tool
```

### Example: trigger RF training

```bash
curl -s -X POST http://localhost:8000/training/start \
  -H "Content-Type: application/json" \
  -d '{"stage": "rf", "data_source": "csv"}' \
  | python3 -m json.tool
# Returns: {"id": 1, "status": "queued", ...}

# Poll status
curl -s http://localhost:8000/training/status/1 | python3 -m json.tool
```

---

## 11. Project Structure

```
netguard-ids/
│
├── app/                          FastAPI application
│   ├── main.py                     App factory — lifespan, router registration
│   ├── config.py                   Settings (reads .env via pydantic-settings)
│   ├── database.py                 SQLAlchemy async engine + session + init_db()
│   ├── routers/
│   │   ├── predict.py              POST /predict/flow
│   │   ├── explain.py              POST /explain/flow
│   │   ├── alerts.py               GET/POST /alerts/
│   │   ├── metrics.py              GET/PUT /metrics/
│   │   ├── stream.py               WS /stream/ws
│   │   └── training.py             POST/GET /training/
│   ├── services/
│   │   ├── preprocessor.py         Feature dict → scaled numpy array
│   │   ├── inference.py            RFInference + AEInference
│   │   ├── rule_engine.py          10 deterministic rules (R-01 … R-10)
│   │   ├── consensus.py            Dual-model consensus → Consensus + Severity
│   │   └── explainer.py            RF global importance + AE per-feature error
│   ├── db/
│   │   ├── models.py               SQLAlchemy ORM tables
│   │   └── crud.py                 Async CRUD + feedback-label bridge
│   └── models/
│       ├── lstm_ae.py              LSTMAutoencoder (encoder + decoder)
│       └── schemas.py              Pydantic request/response schemas
│
├── training/                     Training pipeline
│   ├── pipelines/training_pipeline.py   Orchestrator (run_rf / run_ae_phase1 / run_ae_phase2)
│   ├── trainers/                   RFTrainer, NSLKDDAETrainer, CICIDSAETrainer
│   ├── loaders/                    CICIDS2017Loader, NSLKDDLoader
│   ├── preprocessors/              CICIDSPreprocessor (RobustScaler + feature selector)
│   ├── evaluators/                 ClassificationEvaluator, AnomalyEvaluator
│   ├── train_rf.py                 CLI entry: python -m training.train_rf
│   ├── pretrain_nslkdd.py          CLI entry: python -m training.pretrain_nslkdd
│   ├── finetune_cicids.py          CLI entry: python -m training.finetune_cicids
│   └── export_feedback_dataset.py  CLI: export analyst-labelled training rows
│
├── mcp_server/
│   └── server.py                 FastMCP — predict_flow, list_alerts, get_metrics tools
│
├── ui/
│   ├── templates/index.html      Alpine.js dashboard (Alerts tab + Training tab)
│   └── static/                   CSS + JS
│
├── tests/                        pytest suite (20 tests, SQLite)
│   ├── conftest.py                 Forces SQLite via pytest_configure
│   ├── test_api.py
│   ├── test_inference.py
│   └── test_mcp.py
│
├── models/                       Trained artifacts (gitignored)
├── data/                         Datasets (gitignored)
├── logs/                         Runtime logs: api.log, mcp.log (gitignored)
│
├── Dockerfile                    API + MCP image (Python 3.11-slim)
├── Dockerfile.ui                 UI image (nginx:1.27-alpine)
├── nginx.conf                    Nginx reverse-proxy + static-asset config
├── docker-compose.yml            4-service stack: db · api · ui · mcp
│
├── setup.sh                      One-time environment setup
├── ops.sh                        Start / stop / restart / logs / train
├── requirements.txt              Python dependencies
├── .env                          Runtime config (DATABASE_URL, AE_THRESHOLD, …)
├── DECISIONS.md                  Design log + future extension queue
└── README.md                     This file
```

### Database tables

| Table | Purpose |
|-------|---------|
| `alert_log` | Every prediction result — RF label, AE score, consensus, severity |
| `feedback` | Analyst verdicts (TP / FP / FN) linked to alert_log |
| `feature_log` | Preprocessed feature vector per alert — raw material for retraining |
| `training_run` | Training job history — stage, status, metrics, duration |
| `threshold_log` | Audit trail of AE threshold changes |

---

## 12. Feedback-Label Bridge

Analyst feedback drives incremental retraining. The bridge joins `feature_log`
+ `alert_log` + `feedback` and assigns corrected training labels:

| Analyst verdict | Label assigned |
|-----------------|----------------|
| `FALSE_POSITIVE` | `BENIGN` — analyst says it was safe |
| `TRUE_POSITIVE` | RF-predicted attack class — confirmed |
| `FALSE_NEGATIVE` | `UNKNOWN_ATTACK` — missed attack found by analyst |
| No feedback, RF confidence ≥ 0.95 | RF-predicted class — auto-label |

### Export corrected dataset

```bash
conda activate dl_env
PYTHONPATH=. python -m training.export_feedback_dataset \
    --db sqlite+aiosqlite:///./netguard_dev.db \
    --out data/feedback_dataset.csv
```

For Docker / PostgreSQL:

```bash
PYTHONPATH=. python -m training.export_feedback_dataset \
    --db postgresql+asyncpg://netguard:netguard_dev@127.0.0.1:5432/netguard \
    --out data/feedback_dataset.csv
```
