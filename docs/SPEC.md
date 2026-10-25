# NetGuard IDS — Functional Specification

## 1. Purpose

NetGuard IDS is an individual A2 implementation for CS7003NU Advanced AI
Technologies. It implements the network intrusion detection pipeline designed
in A1, using CIC-IDS2017 as the dataset, and serves the trained models as a
real-life application with a REST API, analyst dashboard UI, and MCP server.

---

## 2. Functional Requirements

### FR-01 Single-flow classification
The system shall accept a single network flow (JSON dict of feature name →
value) via POST /api/predict and return a classification verdict within 500ms.

### FR-02 Batch classification
The system shall accept a CSV file of up to 50,000 flows via POST
/api/predict/batch and return a summary + per-row results.

### FR-03 Three-stage hybrid inference
Every prediction shall pass through a 3-stage pipeline:
- **Stage 1 (Rule Engine):** 10 deterministic rules (traffic thresholds, flag
  anomalies, known malicious ports). If verdict = BENIGN, return SAFE and skip AI.
- **Stage 2 (Random Forest):** Supervised classifier run only when Stage 1 is SUSPICIOUS.
- **Stage 3 (LSTM Autoencoder):** Unsupervised anomaly scorer run only when Stage 1 is SUSPICIOUS.
The consensus engine shall produce one of four verdicts:
SAFE | CONFIRMED | KNOWN_PATTERN | ZERO_DAY

### FR-12 Rule transparency
Every prediction response shall include the full rule engine result:
triggered rule IDs, confidence score, and whether AI was skipped (ai_skipped flag).

### FR-04 Explainability
The system shall provide per-prediction explanations: SHAP values for the RF
(top-10 features) and per-feature reconstruction error for the AE.

### FR-05 Alert persistence
Every detection shall be written to the SQLite alert log with timestamp,
input features, both model outputs, consensus verdict, and severity.

### FR-06 Analyst feedback
An analyst shall be able to mark any alert as TRUE_POSITIVE, FALSE_POSITIVE,
or FALSE_NEGATIVE via POST /api/alerts/{id}/feedback.

### FR-07 Threshold tuning
An analyst shall be able to adjust the AE detection threshold live via POST
/api/threshold. Changes are logged and take immediate effect.

### FR-08 Traffic replay
The UI shall support a replay mode: stream a pre-loaded sample of CIC-IDS2017
test flows as a WebSocket feed at configurable speed (default 200ms/flow),
with alerts appearing live in the dashboard.

### FR-09 Metrics endpoint
The system shall expose precision, recall, F1 and false-positive rate
(computed against analyst feedback) via GET /api/metrics.

### FR-10 MCP server
The system shall expose 6 MCP tools (classify_flow, batch_classify,
explain_detection, tune_sensitivity, query_alerts, generate_incident_report)
so any MCP-capable client (Claude, Codex) can act as a security analyst.

### FR-11 Incident report
An analyst or MCP client shall be able to export a JSON incident report
covering all alerts within a specified time range.

---

## 3. Non-functional Requirements

| NFR | Requirement |
|---|---|
| Performance | Single-flow prediction < 500ms (RF: <10ms, AE: <100ms, SHAP: <400ms) |
| Availability | Runs locally; models loaded at startup and held in memory |
| Portability | Runs on `dl_env` conda env (Python 3.12, CUDA optional, CPU fallback) |
| Scalability | Batch endpoint handles 50k rows via pandas chunking |
| Security | Input validation via Pydantic; no external network calls from model layer |
| Observability | /api/health returns model load status, uptime, threshold |

---

## 4. A2 Marking Criteria Mapping

| Criterion | Weight | How this project addresses it |
|---|---|---|
| a) Functional specification | 10% | This SPEC.md + FR-01 to FR-11 |
| b) Software architecture & components | 10% | DESIGN.md §2 architecture diagram + 7 distinct layers |
| c) Libraries & development tools | 10% | requirements.txt with justified choices (§5 below) |
| d) Configure parameters & test data | 20% | Training scripts (NSL-KDD pretrain, CIC-IDS2017 fine-tune), threshold tuning, RF hyperparameters, CIC-IDS2017 test set |
| e) Demonstrate the program | 20% | Screen recording: UI demo + traffic replay + MCP/Claude integration |
| Report the work | 30% | A2 report with statistical results (F1, AUC, per-class metrics, transfer learning comparison) |

---

## 5. Technology Stack & Rationale

### Backend
| Library | Version | Rationale |
|---|---|---|
| FastAPI | ≥0.115 | Async-first, auto OpenAPI docs, WebSocket support, Pydantic integration |
| Uvicorn | ≥0.30 | ASGI server; supports HTTP + WebSocket in one process |
| SQLAlchemy | ≥2.0 | Async ORM; future-proof if SQLite → Postgres |
| aiosqlite | ≥0.19 | Async SQLite driver for SQLAlchemy |
| python-multipart | ≥0.0.9 | CSV file upload handling |

### Machine Learning
| Library | Version | Rationale |
|---|---|---|
| PyTorch | ≥2.1 | LSTM-AE implementation; CUDA support on RTX A2000 |
| scikit-learn | ≥1.4 | Random Forest; already used in A1 pipeline |
| joblib | ≥1.3 | RF model serialisation (standard sklearn approach) |
| SHAP | ≥0.44 | TreeExplainer for RF; fast, model-native explanations |
| numpy | ≥1.26 | Array operations throughout |
| pandas | ≥2.1 | CSV ingestion, feature pipeline, batch processing |

### MCP Server
| Library | Version | Rationale |
|---|---|---|
| fastmcp | ≥2.0 | Declarative MCP tool definitions with minimal boilerplate |

### Frontend
| Technology | Rationale |
|---|---|
| Vanilla HTML/CSS/JS | No build step; works in any browser; easy to demo |
| Chart.js (CDN) | Lightweight charting for alert timeline + metrics |
| Alpine.js (CDN) | Minimal reactivity for UI state without a framework |

### Training / Data
| Library | Version | Rationale |
|---|---|---|
| requests / urllib | stdlib | NSL-KDD download |
| tqdm | ≥4.66 | Training progress bars |
| matplotlib | ≥3.8 | Training loss curves |

---

## 6. Data Flow — Single Prediction (3-Stage Pipeline)

```
Analyst submits flow (JSON)
    │
    ▼
Pydantic validates + normalises field names
    │
    ▼
Stage 1 — RuleEngine.evaluate(features)   [non-AI, deterministic]
  10 rules: rate thresholds, flag anomalies, known malicious ports
    │
    ├── BENIGN ──────────────────────────────────────────────────┐
    │                                                            ▼
    │                                               consensus = SAFE
    │                                               ai_skipped = True
    │                                               CRUD.insert_alert() → SQLite
    │                                               JSON response to client
    │
    └── SUSPICIOUS
          │
          ▼
    Preprocessor.transform()
      ├── apply saved StandardScaler
      └── apply saved feature selector
          │
          ▼
Stage 2 — RF.predict_proba()              [supervised AI]
          → class label + confidence
          → SHAP TreeExplainer
          │
          ▼
Stage 3 — LSTM-AE.encode→decode()         [unsupervised AI]
          → reconstruction error
          → vs threshold θ → BENIGN | ANOMALY
          │
          ▼
    ConsensusEngine.decide(rf, ae)
    → verdict + severity
          │
          ▼
    CRUD.insert_alert() → SQLite
          │
          ▼
    JSON response to client
```

---

## 7. Replay Mode Data Flow

```
GET /api/stream  (WebSocket upgrade)
    │
    ▼
Server loads replay_sample.csv (500 rows from CIC-IDS2017 test set)
    │
    ▼
For each row (every 200ms):
  run prediction → consensus verdict
  push JSON event over WebSocket
    │
    ▼
Browser receives event → renders alert card + updates chart
```
