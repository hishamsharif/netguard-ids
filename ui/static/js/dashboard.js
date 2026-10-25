const CSV_HEADER = "Destination Port,Flow Duration,Total Fwd Packets,Total Backward Packets,Total Length of Fwd Packets,Total Length of Bwd Packets,Fwd Packet Length Max,Fwd Packet Length Min,Fwd Packet Length Mean,Fwd Packet Length Std,Bwd Packet Length Max,Bwd Packet Length Min,Bwd Packet Length Mean,Bwd Packet Length Std,Flow Bytes/s,Flow Packets/s,Flow IAT Mean,Flow IAT Std,Flow IAT Max,Flow IAT Min";
const BENIGN_ROW  = "443,50000,10,8,500,400,100,20,50.0,15.0,80,10,50.0,12.0,10000.0,200.0,5000.0,1000.0,10000,100";
const ATTACK_ROW  = "22,100,500,0,0,0,0,0,0.0,0.0,0,0,0.0,0.0,0.0,5000000.0,0.2,0.0,1,0";

function dashboard() {
  return {
    connected: false,
    replaying: false,
    alerts: [],
    metrics: {},
    selectedAlert: null,
    explanation: {},
    ws: null,
    activeTab: 'infer',

    feedbackState: {},

    // 'paste' | 'upload'
    inferMode: 'paste',

    // Model versions registry
    modelVersions: { versions: [], active: null, pending_logs: 0, total_inference_logs: 0, active_rf_id: null, active_ae_id: null },

    // Inference state (replaces old predict + drives both paste/upload modes)
    infer: {
      csvText: CSV_HEADER + "\n" + ATTACK_ROW,
      rfVersionId: null,
      aeVersionId: null,
      applyMsg: '',
    },

    // Batch CSV result state (shared between paste and upload)
    batch: {
      loading: false,
      result: null,
      error: null,
      dragOver: false,
    },

    // Training panel state
    training: {
      stage: 'all',
      dataSource: 'csv',
      running: false,
      currentRun: null,
      history: [],
      pollTimer: null,
    },

    async init() {
      await this.loadAlerts();
      await this.loadMetrics();
      await this.loadModelVersions();
      setInterval(() => this.loadMetrics(), 30000);
    },

    async loadAlerts() {
      const res = await fetch('/alerts/?limit=50');
      this.alerts = await res.json();
      if (window.updateCharts) updateCharts(this.alerts, this.metrics.ae_threshold);
    },

    async loadMetrics() {
      const res = await fetch('/metrics/');
      this.metrics = await res.json();
      if (window.updateCharts) updateCharts(this.alerts, this.metrics.ae_threshold);
    },

    // ── Preset CSV generators ─────────────────────────────────────────────────

    loadPreset(type) {
      const row = type === 'benign' ? BENIGN_ROW : ATTACK_ROW;
      this.infer.csvText = CSV_HEADER + "\n" + row;
    },

    // ── Paste mode: convert text → File → reuse _uploadFile ──────────────────

    async submitCsvPaste() {
      const text = this.infer.csvText.trim();
      if (!text) {
        this.batch.error = 'Paste CSV content first';
        return;
      }
      const blob = new Blob([text], { type: 'text/csv' });
      const file = new File([blob], 'paste.csv', { type: 'text/csv' });
      await this._uploadFile(file);
    },

    // ── Batch CSV upload core (shared) ────────────────────────────────────────

    handleDrop(ev) {
      ev.preventDefault();
      this.batch.dragOver = false;
      const file = ev.dataTransfer?.files[0];
      if (file) this._uploadFile(file);
    },

    handleFileInput(ev) {
      const file = ev.target?.files[0];
      if (file) this._uploadFile(file);
    },

    async _uploadFile(file) {
      if (!file.name.endsWith('.csv')) {
        this.batch.error = 'Please upload a .csv file';
        return;
      }
      this.batch.loading = true;
      this.batch.error = null;
      this.batch.result = null;
      const fd = new FormData();
      fd.append('file', file);
      try {
        const res = await fetch('/predict/batch', { method: 'POST', body: fd });
        if (!res.ok) {
          const err = await res.json();
          this.batch.error = err.detail || `HTTP ${res.status}`;
          return;
        }
        this.batch.result = await res.json();
        await this.loadAlerts();
      } catch (e) {
        this.batch.error = 'Upload error: ' + e.message;
      } finally {
        this.batch.loading = false;
      }
    },

    batchConsensusSummary() {
      if (!this.batch.result?.consensus_counts) return [];
      return Object.entries(this.batch.result.consensus_counts).map(([k, v]) => ({ label: k, count: v }));
    },

    // ── Alert explain via stored raw_features ─────────────────────────────────

    async selectAlert(alert) {
      this.selectedAlert = alert;
      this.explanation = { natural_language: 'Loading explanation…', loading: true };
      if (window.renderImportanceChart) renderImportanceChart([]);
      if (window.renderAeErrorChart) renderAeErrorChart([]);
      try {
        const detailRes = await fetch(`/alerts/${alert.id}/detail`);
        if (!detailRes.ok) throw new Error(`Detail fetch failed: ${detailRes.status}`);
        const detail = await detailRes.json();
        if (!detail.raw_features) {
          this.explanation = { natural_language: 'No stored features for this alert. Run a new flow and explain it.' };
          return;
        }
        const explainRes = await fetch('/explain/flow', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ features: detail.raw_features }),
        });
        if (!explainRes.ok) throw new Error(`Explain failed: ${explainRes.status}`);
        const data = await explainRes.json();
        this.explanation = data;
        if (window.renderImportanceChart) renderImportanceChart(data.rf_importance || []);
        if (window.renderAeErrorChart) renderAeErrorChart(data.ae_recon_error || []);
        if (detail.feedback?.length) {
          this.feedbackState[alert.id] = detail.feedback[detail.feedback.length - 1].verdict;
        }
      } catch (e) {
        this.explanation = { natural_language: 'Explain error: ' + e.message };
      }
    },

    // ── Explain from a batch result row (uses alert_id stored in result) ───────

    async explainBatchRow(row) {
      if (!row.alert_id) {
        this.explanation = { natural_language: 'No alert_id returned for this row — cannot explain.' };
        this.selectedAlert = { id: 'batch-row', consensus: row.consensus };
        return;
      }
      const fakeAlert = { id: row.alert_id, consensus: row.consensus, severity: row.severity };
      await this.selectAlert(fakeAlert);
    },

    // ── Feedback ──────────────────────────────────────────────────────────────

    async submitFeedback(alertId, verdict) {
      this.feedbackState[alertId] = 'pending';
      try {
        await fetch(`/alerts/${alertId}/feedback`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ verdict }),
        });
        this.feedbackState[alertId] = verdict;
      } catch (e) {
        delete this.feedbackState[alertId];
        alert('Feedback error: ' + e.message);
      }
    },

    feedbackLabel(alertId) {
      const v = this.feedbackState[alertId];
      if (!v) return null;
      if (v === 'pending') return '…';
      return { TRUE_POSITIVE: '✓ TP', FALSE_POSITIVE: '✗ FP', FALSE_NEGATIVE: '? FN' }[v] || v;
    },

    consensusColorClass(val) {
      return { SAFE: 'tag-safe', CONFIRMED: 'tag-confirmed', KNOWN_PATTERN: 'tag-known_pattern', ZERO_DAY: 'tag-zero_day' }[val] || '';
    },

    // ── Model version registry ────────────────────────────────────────────────

    async loadModelVersions() {
      try {
        const res = await fetch('/models/');
        const data = await res.json();
        this.modelVersions = {
          versions: data.versions || [],
          active: data.versions?.find(v => v.is_active) || null,
          pending_logs: data.pending_logs || 0,
          total_inference_logs: data.total_inference_logs || 0,
          active_rf_id: data.active_rf_id || null,
          active_ae_id: data.active_ae_id || null,
        };
        // Default inference selectors to current active versions
        if (!this.infer.rfVersionId) {
          this.infer.rfVersionId = data.active_rf_id || (data.versions?.[0]?.id ?? null);
        }
        if (!this.infer.aeVersionId) {
          this.infer.aeVersionId = data.active_ae_id || (data.versions?.[0]?.id ?? null);
        }
      } catch (_) {}
    },

    // Computed filter: versions that have an RF model
    rfVersions() {
      return (this.modelVersions.versions || []).filter(v => v.model_path_rf);
    },

    // Computed filter: versions that have an AE model
    aeVersions() {
      return (this.modelVersions.versions || []).filter(v => v.model_path_ae);
    },

    async applyModelVersions() {
      this.infer.applyMsg = '';
      const results = [];
      if (this.infer.rfVersionId) {
        try {
          await fetch(`/models/${this.infer.rfVersionId}/activate/rf`, { method: 'POST' });
          results.push('RF activated');
        } catch (_) {
          results.push('RF error');
        }
      }
      if (this.infer.aeVersionId) {
        try {
          await fetch(`/models/${this.infer.aeVersionId}/activate/ae`, { method: 'POST' });
          results.push('AE activated');
        } catch (_) {
          results.push('AE error');
        }
      }
      await this.loadModelVersions();
      this.infer.applyMsg = results.join(' · ') + ' — takes effect on next restart';
    },

    async activateRfVersion(versionId) {
      try {
        const res = await fetch(`/models/${versionId}/activate/rf`, { method: 'POST' });
        const data = await res.json();
        await this.loadModelVersions();
        alert(`RF activated: ${data.active_rf?.version_tag}\n\n${data.note}`);
      } catch (e) {
        alert('Activate RF error: ' + e.message);
      }
    },

    async activateAeVersion(versionId) {
      try {
        const res = await fetch(`/models/${versionId}/activate/ae`, { method: 'POST' });
        const data = await res.json();
        await this.loadModelVersions();
        alert(`AE activated: ${data.active_ae?.version_tag}\n\n${data.note}`);
      } catch (e) {
        alert('Activate AE error: ' + e.message);
      }
    },

    // ── Replay ────────────────────────────────────────────────────────────────

    toggleReplay() {
      if (this.replaying) { this.stopReplay(); } else { this.startReplay(); }
    },

    startReplay() {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      this.ws = new WebSocket(`${proto}://${location.host}/stream/ws`);
      this.ws.onopen = () => { this.connected = true; this.replaying = true; };
      this.ws.onclose = () => { this.connected = false; this.replaying = false; };
      this.ws.onmessage = (ev) => {
        const row = JSON.parse(ev.data);
        this.alerts.unshift(row);
        if (this.alerts.length > 200) this.alerts.pop();
        if (window.pushTimelinePoint) pushTimelinePoint(row);
      };
    },

    stopReplay() {
      if (this.ws) this.ws.close();
      this.replaying = false;
    },

    // ── Training ──────────────────────────────────────────────────────────────

    async startTraining() {
      if (this.training.running) return;
      this.training.running = true;
      this.training.currentRun = null;
      try {
        const res = await fetch('/training/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ stage: this.training.stage, data_source: this.training.dataSource }),
        });
        if (!res.ok) {
          const err = await res.json();
          alert('Training error: ' + (err.detail || res.status));
          this.training.running = false;
          return;
        }
        const run = await res.json();
        this.training.currentRun = run;
        this._pollTrainingStatus(run.id);
      } catch (e) {
        alert('Failed to start training: ' + e.message);
        this.training.running = false;
      }
    },

    async _pollTrainingStatus(runId) {
      if (this.training.pollTimer) clearInterval(this.training.pollTimer);
      this.training.pollTimer = setInterval(async () => {
        try {
          const res = await fetch(`/training/status/${runId}`);
          const run = await res.json();
          this.training.currentRun = run;
          if (run.status === 'done' || run.status === 'failed') {
            clearInterval(this.training.pollTimer);
            this.training.running = false;
            await this.loadTrainingHistory();
            await this.loadModelVersions();
          }
        } catch (_) {}
      }, 3000);
    },

    async loadTrainingHistory() {
      try {
        const res = await fetch('/training/history?limit=10');
        this.training.history = await res.json();
      } catch (_) {}
    },

    trainingStatusClass(status) {
      return { queued: 'tag-known_pattern', running: 'tag-zero_day', done: 'tag-safe', failed: 'tag-confirmed' }[status] || '';
    },
  };
}
