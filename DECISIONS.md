# NetGuard IDS — Design & Implementation Decision Log

Append-only log of architectural decisions, implementation challenges, and items
queued for future extension.  Format per entry: date, title, context, decision,
rationale, trade-offs, and status.

---

## 2026-10 — Architecture

### D-001 · Dual-model detection pipeline (RF + LSTM-AE)
**Decision:** Use a three-stage pipeline: Rule Engine → Random Forest → LSTM Autoencoder → Consensus.  
**Rationale:** RF excels at known-attack classification (high-confidence labelling); LSTM-AE catches
anomalies and zero-days the RF has never seen. The rule engine provides a fast deterministic bypass
for obviously benign flows, saving GPU cycles.  
**Trade-off:** Higher inference latency than a single model; requires two training pipelines.  
**Status:** Implemented.

### D-002 · Transfer learning: NSL-KDD → CIC-IDS2017
**Decision:** Pretrain LSTM-AE on NSL-KDD normal traffic (Phase 1), then fine-tune on CIC-IDS2017
benign flows (Phase 2) with encoder frozen for the first 10 epochs.  
**Rationale:** NSL-KDD is a clean, balanced dataset; pretraining gives the encoder a good
representation of "normal" network behaviour before adapting to the noisier CIC-IDS2017 distribution.  
**Challenge:** Feature dimension mismatch (NSL-KDD: 39 features, CIC-IDS2017: 70). Resolved via
partial weight transfer — copied 15 shape-compatible tensors (hh-weights, biases, decoder LSTM)
and re-initialised the 3 mismatched ones (`weight_ih_l0`, `decoder.fc.weight/bias`).  
**Status:** Implemented. Phase 1 threshold = 1.9415; Phase 2 threshold = 0.008927.

### D-003 · IQR=0 clipping fix for CIC-IDS2017 sparse features
**Decision:** Apply `np.clip(X, -10, 10)` after RobustScaler in both training and inference.  
**Context:** CIC-IDS2017 has sparse flag-count features (0 for ~99% of flows). RobustScaler sets
scale=1 when IQR=0, leaving raw values up to 10^6+ in the LSTM input, producing training loss of
~8.7 trillion.  
**Constant:** `AE_CLIP = 10.0` defined in `training_pipeline.py`; `_CLIP = 10.0` in `inference.py`.
Must remain in sync.  
**Status:** Implemented.

### D-004 · SQLite (dev) → PostgreSQL (production)
**Decision:** Use PostgreSQL with asyncpg as the production database; tests use SQLite via
`tests/conftest.py` `pytest_configure` override to avoid requiring a running PG instance in CI.  
**Rationale:** PostgreSQL supports concurrent writes from multiple API workers, proper JSONB for
feature vectors in future, and row-level locking needed for a real SOC deployment.  
**Status:** Implemented. DB URL in `.env`; tables auto-created on startup via `init_db()`.

### D-005 · FeatureLog + feedback-label bridge
**Decision:** Store preprocessed (scaled + clipped) feature vectors in `feature_log` alongside each
alert, enabling incremental retraining from analyst feedback.  
**Label priority:** FALSE_POSITIVE → BENIGN; TRUE_POSITIVE → rf_label; FALSE_NEGATIVE →
UNKNOWN_ATTACK; no feedback with rf_confidence ≥ 0.95 → rf_label (auto-label).  
**Status:** Implemented. CLI export: `python -m training.export_feedback_dataset`.

### D-006 · Training API background task approach (asyncio vs task queue)
**Decision:** Use `asyncio.create_task` + `run_in_executor` for the training trigger (MVP scope).  
**Limitation:** If the API process restarts mid-training, the job is silently lost — the DB row
remains stuck at `status=running`. This is acceptable for a single-server dev/coursework deployment.  
**Future extension (FE-001):** Replace with Celery + Redis (or RQ) for durable job queuing.  
**Status:** Implemented as asyncio. Durability caveat documented.

### D-007 · UI training panel — partial control model
**Decision:** The UI can trigger and monitor training, but the CLI is the authoritative path for
production training runs. The UI calls `POST /training/start` and polls `GET /training/status/{id}`
every 3 seconds.  
**Rationale:** Browser sessions are ephemeral; a long training run (40+ min) should not depend on
an open tab. The CLI always works regardless of UI state.  
**Status:** Implemented.

### D-008 · Docker service split (4 containers)
**Decision:** Split into four services: `db` (PostgreSQL), `api` (FastAPI), `ui` (Nginx), `mcp` (FastMCP).  
**Rationale:** Separation of concerns; Nginx serves static assets without touching the Python process;
MCP server is an optional add-on that restarts independently.  
**Status:** Implemented. `docker-compose.yml` + `Dockerfile.ui` + `nginx.conf` created.

### D-010 · SQLite local dev / PostgreSQL Docker split
**Decision:** `.env` defaults to `sqlite+aiosqlite:///./netguard_dev.db` for local development.
`docker-compose.yml` overrides `DATABASE_URL` to PostgreSQL via its `environment:` block.
PostgreSQL local setup (`bash setup.sh --with-pg`) is opt-in, not required.  
**Rationale:** Dataset DW tables (FE-002) are not yet built, so PostgreSQL offers no advantage
over SQLite for the current operational tables (AlertLog, Feedback, FeatureLog, TrainingRun).
SQLite needs zero setup and is already async via aiosqlite. PostgreSQL becomes the right choice
when multi-worker concurrency or DW analytics (FE-003) is added.  
**Status:** Implemented.

---

## RF Training Results (2026-10-24)

| Metric | Value |
|---|---|
| Macro F1 | 0.887 |
| Accuracy | 99.86% |
| Training time | 217.9s (20 cores, 200 trees) |
| Weak classes | Web Attack SQL Injection F1=0.40 (3 test samples), XSS F1=0.52 (98 samples) |

Flow-level features cannot distinguish web-layer (L7) attacks well; low F1 on those classes
is expected and does not indicate a model defect.

## AE Training Results (2026-10-24)

| Stage | Dataset | Epochs | Final val_loss | Threshold | Time |
|---|---|---|---|---|---|
| Phase 1 (pretrain) | NSL-KDD normal | 30 | — | 1.9415 | 47s (GPU) |
| Phase 2 (fine-tune) | CIC-IDS2017 benign (2.27M rows) | 20 | 0.002094 | 0.008927 | 2611.4s (~43m, GPU) |

---

## Future Extensions Queue

| ID | Item | Complexity | Priority |
|---|---|---|---|
| FE-001 | Replace `asyncio.create_task` training with Celery + Redis for job durability across restarts | Medium | High |
| FE-002 | **PostgreSQL DW integration** — promote PostgreSQL from Docker-only to the primary DB once dataset tables are built; add `scripts/import_datasets.py` to populate CIC-IDS2017 / NSL-KDD tables; unlock `data_source=db` in the training API; migrate `.env` default from SQLite to PostgreSQL | High | Medium |
| FE-003 | DuckDB analytics layer — once PostgreSQL DW is live (FE-002), attach DuckDB as a read-only query engine over feature_log and dataset tables for fast aggregations without hitting the operational DB; DuckDB can query PostgreSQL directly via its postgres scanner extension, requiring no separate ETL | Low | Low |
| FE-004 | Incremental retraining trigger — auto-queue a new RF/AE run when feedback_dataset exceeds N rows | Medium | Medium |
| FE-005 | AE threshold auto-update after Phase 2 completion (write back to `.env` from pipeline output) | Low | Low |
| FE-006 | SHAP explanation panel — wire up actual stored features from feature_log for per-alert SHAP | Medium | High |
| FE-007 | Multi-worker API (uvicorn + gunicorn) with connection pooling (asyncpg pool size tuning) | Low | Medium |
| FE-008 | CI pipeline — GitHub Actions running tests against a Docker PG service container | Low | High |
| FE-009 | Model versioning — save each training run's model artifact with run_id so rollback is possible | Medium | Medium |
| FE-010 | Live threshold recalibration endpoint — recompute p95 threshold from recent feature_log rows | Medium | Low |
| FE-011 | Log WebSocket-streamed flows to alert_log + feature_log (currently stream.py runs inference but does not persist results) | Low | Medium |
| FE-012 | Replace global RF feature importances in explain endpoint with per-sample SHAP values wired to feature_log (needs re-adding shap>=0.44.0 to requirements) | Medium | Low |
