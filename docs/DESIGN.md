# NetGuard IDS — Design Document

## 1. System Overview

NetGuard IDS is a real-time Network Intrusion Detection System built for
security operations centre (SOC) analysts. It accepts network flow records
(CIC-IDS2017 format, 79 features per flow), classifies them using a
**3-stage hybrid pipeline**, surfaces explanations, and exposes an MCP server
so that Claude or any MCP-capable AI assistant can act as a conversational
security analyst.

### 3-Stage Detection Pipeline

```
Stage 1 ── Rule Engine (deterministic, non-AI)
              │
              ├── BENIGN → SAFE  (AI skipped — fast path)
              │
              └── SUSPICIOUS
                      │
            Stage 2 ── Random Forest (supervised AI)
                      │   → attack type + confidence + SHAP
                      │
            Stage 3 ── LSTM Autoencoder (unsupervised AI)
                      │   → reconstruction error vs threshold
                      │
                      └── Consensus Engine
                              SAFE | CONFIRMED | KNOWN_PATTERN | ZERO_DAY
```

**Why this order:**
- Rule Engine catches obvious known-signature attacks cheaply (no ML inference cost).
- Random Forest classifies the attack type for known labelled patterns.
- LSTM-AE detects novel/zero-day anomalies RF may miss (no label needed).
- Flows the rule engine clears as benign never touch the GPU/CPU-heavy models.

**Domain:** Pattern Recognition → Network Intrusion Detection  
**Dataset:** CIC-IDS2017 (2.8M flows, 79 features, 15 attack types + benign)  
**A1 link:** same problem, same dataset, now implemented and served  

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          UI Layer                               │
│  Analyst Dashboard (vanilla JS + Chart.js)                      │
│  ├── Manual flow entry form                                     │
│  ├── CSV batch upload → labelled results table                  │
│  ├── Traffic replay (WebSocket streaming at 200ms/flow)         │
│  ├── Dual-model consensus panel (RF vs LSTM-AE agreement)       │
│  ├── SHAP feature importance chart                              │
│  ├── Alert feed (live + historical)                             │
│  ├── AE threshold tuning slider                                 │
│  └── Incident report export (JSON)                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP / WebSocket
┌───────────────────────────▼─────────────────────────────────────┐
│                      FastAPI REST Layer                         │
│  POST /api/predict              single flow → verdict           │
│  POST /api/predict/batch        CSV upload → bulk results       │
│  GET  /api/explain/{flow_id}    SHAP + reconstruction error     │
│  POST /api/threshold            tune AE detection threshold     │
│  GET  /api/alerts               alert log (paginated)           │
│  POST /api/alerts/{id}/feedback analyst marks FP / FN           │
│  GET  /api/metrics              live precision / recall / F1    │
│  WS   /api/stream               WebSocket — replay mode         │
│  GET  /api/health               model status + uptime           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                       Service Layer                             │
│  RuleEngine (Stage 1) — deterministic heuristics (10 rules)     │
│    R-01 DoS rate · R-02 SYN flood · R-03 RST flood              │
│    R-04 FIN scan  · R-05 URG flag · R-06 high-risk port         │
│    R-07 port probe· R-08 null session · R-09 oversized payload  │
│    R-10 PSH+ACK flood                                           │
│  InferenceService (Stage 2+3) — RF + LSTM-AE (gated by Stage 1) │
│  ConsensusEngine    — combines verdicts, flags zero-day zone    │
│  Explainer          — SHAP (RF) + per-feature recon error (AE)  │
│  Preprocessor       — mirrors A1 pipeline (scaler, selector)    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                       Model Layer                               │
│  Random Forest (joblib)          LSTM Autoencoder (PyTorch)     │
│  ├── Multi-class: 16 labels      ├── Pretrained: NSL-KDD        │
│  ├── Feature importance (SHAP)   ├── Fine-tuned: CIC-IDS2017    │
│  └── Probability per class       └── Anomaly: reconstruction ε  │
│                                                                 │
│  Consensus Engine                                               │
│  RF=benign + AE=benign  → ✅ SAFE                               │
│  RF=attack + AE=attack  → 🔴 CONFIRMED THREAT                   │
│  RF=attack + AE=benign  → 🟡 KNOWN PATTERN                      │
│  RF=benign + AE=attack  → 🟠 POTENTIAL ZERO-DAY                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Persistence Layer                            │
│  SQLite (aiosqlite + SQLAlchemy async)                          │
│  ├── alert_log       — every detection with timestamp + verdict │
│  ├── feedback        — analyst FP/FN corrections                │
│  └── threshold_log   — AE threshold change history             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      MCP Server (FastMCP)                       │
│  Tool: classify_flow(features)       → verdict + explanation    │
│  Tool: batch_classify(csv_b64)       → threat summary           │
│  Tool: explain_detection(flow_id)    → natural-lang breakdown   │
│  Tool: tune_sensitivity(threshold)   → adjust AE threshold      │
│  Tool: query_alerts(type, n)         → recent alerts by type    │
│  Tool: generate_incident_report()    → session JSON summary     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Transfer Learning Strategy

### Model: LSTM Autoencoder

Architecture based on Malhotra et al. (2016) *"LSTM-based Encoder-Decoder
for Multi-sensor Anomaly Detection"*.

```
Input sequence  [B, T, F]   B=batch, T=window(10 flows), F=features
      │
  LSTM Encoder  (2 layers, hidden=128)
      │  → context vector [B, 128]
  LSTM Decoder  (2 layers, hidden=128)
      │  → reconstructed sequence [B, T, F]
      │
  Anomaly score = mean squared reconstruction error per flow
  Threshold θ   = 95th-percentile error on normal validation set
```

### Transfer learning phases

**Phase 1 — Pretrain on NSL-KDD**
- NSL-KDD is the canonical, well-cited IDS benchmark (Tavallaee et al. 2009)
- 125,973 flows, 41 features → encode to 79-dim (zero-pad / project)
- Train LSTM-AE on normal flows only (unsupervised)
- Saves: `models/lstm_ae_nslkdd.pt`

**Phase 2 — Fine-tune on CIC-IDS2017**
- Freeze encoder layers 1–N, unfreeze decoder + last encoder layer
- Fine-tune on CIC-IDS2017 normal flows (Monday file — 529,918 benign)
- Saves: `models/lstm_ae_cicids.pt`

**Transfer story (report):**
> The LSTM-AE was pretrained on NSL-KDD to learn general representations of
> normal network behaviour, then fine-tuned on CIC-IDS2017 to adapt to modern
> traffic patterns. Freezing the encoder preserves the generalisable low-level
> feature abstractions; only the decoder is retrained to reconstruct the
> higher-dimensional CIC-IDS2017 feature space.

---

## 4. Database Schema

```sql
CREATE TABLE alert_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
    flow_json   TEXT,           -- raw input features as JSON
    rf_label    TEXT,           -- RF prediction (attack type or BENIGN)
    rf_conf     REAL,           -- RF confidence [0,1]
    ae_score    REAL,           -- AE reconstruction error
    ae_label    TEXT,           -- BENIGN or ANOMALY
    consensus   TEXT,           -- SAFE | CONFIRMED | KNOWN_PATTERN | ZERO_DAY
    severity    TEXT            -- LOW | MEDIUM | HIGH | CRITICAL
);

CREATE TABLE feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id    INTEGER REFERENCES alert_log(id),
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
    analyst     TEXT DEFAULT 'analyst',
    verdict     TEXT,           -- TRUE_POSITIVE | FALSE_POSITIVE | FALSE_NEGATIVE
    notes       TEXT
);

CREATE TABLE threshold_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
    old_value   REAL,
    new_value   REAL,
    set_by      TEXT DEFAULT 'analyst'
);
```

---

## 5. API Contract

### POST /api/predict
```json
Request:  { "features": { "Flow Duration": 123, "Total Fwd Packets": 5, ... } }
Response: {
  "flow_id": "uuid",
  "rf": { "label": "DoS Hulk", "confidence": 0.97, "top_class": "DoS Hulk" },
  "ae": { "score": 0.041, "threshold": 0.023, "label": "ANOMALY" },
  "consensus": "CONFIRMED",
  "severity": "HIGH",
  "timestamp": "2026-10-24T10:00:00Z"
}
```

### POST /api/predict/batch
```
Request:  multipart/form-data  file=<csv>
Response: {
  "total": 1000, "safe": 842, "confirmed": 134, "zero_day": 12, "known": 12,
  "attack_breakdown": { "DoS Hulk": 89, "PortScan": 45, ... },
  "rows": [ { flow_id, consensus, severity, rf_label, ae_score }, ... ]
}
```

### GET /api/explain/{flow_id}
```json
{
  "flow_id": "uuid",
  "rf_shap": [ { "feature": "Flow Duration", "shap_value": 0.34 }, ... ],
  "ae_recon_error": [ { "feature": "Total Fwd Packets", "error": 0.12 }, ... ],
  "natural_language": "This flow was flagged primarily because Flow Duration..."
}
```

---

## 6. MCP Tools

| Tool | Parameters | Returns |
|---|---|---|
| `classify_flow` | `features: dict` | verdict + explanation text |
| `batch_classify` | `csv_base64: str` | threat summary stats |
| `explain_detection` | `flow_id: str` | NL feature breakdown |
| `tune_sensitivity` | `threshold: float` | new threshold + impact |
| `query_alerts` | `attack_type: str, n: int` | list of recent alerts |
| `generate_incident_report` | `since: str (ISO)` | JSON incident summary |

---

## 7. Folder Structure

```
netguard-ids/
├── app/
│   ├── main.py              # FastAPI app + lifespan
│   ├── config.py            # Settings (env-based)
│   ├── database.py          # SQLAlchemy async engine
│   ├── models/
│   │   ├── lstm_ae.py       # PyTorch LSTM-AE class
│   │   └── schemas.py       # Pydantic request/response models
│   ├── routers/
│   │   ├── predict.py       # /predict, /predict/batch
│   │   ├── explain.py       # /explain/{flow_id}
│   │   ├── alerts.py        # /alerts, /alerts/{id}/feedback
│   │   ├── stream.py        # WebSocket /stream
│   │   └── metrics.py       # /metrics, /health, /threshold
│   ├── services/
│   │   ├── inference.py     # Model loading + RF/AE inference
│   │   ├── consensus.py     # Consensus engine
│   │   ├── explainer.py     # SHAP + recon error breakdown
│   │   └── preprocessor.py  # Feature pipeline
│   └── db/
│       └── crud.py          # Async CRUD operations
├── mcp_server/
│   └── server.py            # FastMCP tool definitions
├── training/
│   ├── pretrain_nslkdd.py   # Phase 1: NSL-KDD pretraining
│   ├── finetune_cicids.py   # Phase 2: CIC-IDS2017 fine-tuning
│   └── train_rf.py          # RF training + serialisation
├── ui/
│   ├── index.html           # Analyst dashboard
│   └── static/
│       ├── css/main.css
│       └── js/
│           ├── dashboard.js
│           ├── replay.js
│           └── charts.js
├── models/                  # Serialised model artefacts (.pt, .joblib)
├── data/
│   └── nslkdd/              # NSL-KDD download target
├── tests/
│   ├── test_api.py
│   ├── test_inference.py
│   └── test_mcp.py
├── notebooks/
│   └── a2_exploration.ipynb
├── DESIGN.md                # This file
├── SPEC.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── .env.example
```
