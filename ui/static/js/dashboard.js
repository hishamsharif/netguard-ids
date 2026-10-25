const BENIGN_PRESET = {"Destination Port":443,"Flow Duration":50000,"Total Fwd Packets":10,"Total Backward Packets":8,"Total Length of Fwd Packets":500,"Total Length of Bwd Packets":400,"Fwd Packet Length Max":100,"Fwd Packet Length Min":20,"Fwd Packet Length Mean":50.0,"Fwd Packet Length Std":15.0,"Bwd Packet Length Max":80,"Bwd Packet Length Min":10,"Bwd Packet Length Mean":50.0,"Bwd Packet Length Std":12.0,"Flow Bytes/s":10000.0,"Flow Packets/s":200.0,"Flow IAT Mean":5000.0,"Flow IAT Std":1000.0,"Flow IAT Max":10000,"Flow IAT Min":100};
const ATTACK_PRESET = {"Destination Port":22,"Flow Duration":100,"Total Fwd Packets":500,"Total Backward Packets":0,"Total Length of Fwd Packets":0,"Total Length of Bwd Packets":0,"Fwd Packet Length Max":0,"Fwd Packet Length Min":0,"Fwd Packet Length Mean":0.0,"Fwd Packet Length Std":0.0,"Bwd Packet Length Max":0,"Bwd Packet Length Min":0,"Bwd Packet Length Mean":0.0,"Bwd Packet Length Std":0.0,"Flow Bytes/s":0.0,"Flow Packets/s":5000000.0,"Flow IAT Mean":0.2,"Flow IAT Std":0.0,"Flow IAT Max":1,"Flow IAT Min":0};

function dashboard() {
  return {
    connected: false,
    replaying: false,
    alerts: [],
    metrics: {},
    selectedAlert: null,
    explanation: {},
    ws: null,
    activeTab: 'alerts',

    // Per-alert feedback state: { alertId: 'TRUE_POSITIVE' | 'FALSE_POSITIVE' | 'FALSE_NEGATIVE' | 'pending' }
    feedbackState: {},

    // Predict panel state
    predict: {
      featuresJson: JSON.stringify(ATTACK_PRESET, null, 2),
      loading: false,
      error: null,
      result: null,
      lastFeatures: null,
    },

    // Batch CSV upload state
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
      await this.loadTrainingHistory();
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

    // ── Alert explain via stored raw_features ────────────────────────────────

    async selectAlert(alert) {
      this.selectedAlert = alert;
      this.explanation = { natural_language: 'Loading explanation…', loading: true };
      if (window.renderImportanceChart) renderImportanceChart([]);
      if (window.renderAeErrorChart) renderAeErrorChart([]);

      try {
        // Fetch full alert detail (includes raw_features stored at inference time)
        const detailRes = await fetch(`/alerts/${alert.id}/detail`);
        if (!detailRes.ok) throw new Error(`Detail fetch failed: ${detailRes.status}`);
        const detail = await detailRes.json();

        if (!detail.raw_features) {
          this.explanation = { natural_language: 'No stored features for this alert. Use the Predict tab to run a new flow and explain it.' };
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

        // Restore any previously-given feedback for this alert
        if (detail.feedback?.length) {
          this.feedbackState[alert.id] = detail.feedback[detail.feedback.length - 1].verdict;
        }
      } catch (e) {
        this.explanation = { natural_language: 'Explain error: ' + e.message };
      }
    },

    // ── Feedback ─────────────────────────────────────────────────────────────

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

    // ── Predict single flow ───────────────────────────────────────────────────

    loadPreset(type) {
      this.predict.featuresJson = JSON.stringify(type === 'benign' ? BENIGN_PRESET : ATTACK_PRESET, null, 2);
    },

    async submitFlow() {
      this.predict.loading = true;
      this.predict.error = null;
      this.predict.result = null;
      try {
        const features = JSON.parse(this.predict.featuresJson);
        const res = await fetch('/predict/flow', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ features }),
        });
        if (!res.ok) {
          const err = await res.json();
          this.predict.error = err.detail || `HTTP ${res.status}`;
          return;
        }
        this.predict.result = await res.json();
        this.predict.lastFeatures = features;
        await this.loadAlerts();
      } catch (e) {
        this.predict.error = 'JSON parse error: ' + e.message;
      } finally {
        this.predict.loading = false;
      }
    },

    async explainLastFlow() {
      if (!this.predict.lastFeatures) return;
      this.explanation = { natural_language: 'Loading explanation…' };
      this.selectedAlert = { id: 'submitted', consensus: this.predict.result?.consensus };
      if (window.renderImportanceChart) renderImportanceChart([]);
      if (window.renderAeErrorChart) renderAeErrorChart([]);
      try {
        const res = await fetch('/explain/flow', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ features: this.predict.lastFeatures }),
        });
        const data = await res.json();
        this.explanation = data;
        if (window.renderImportanceChart) renderImportanceChart(data.rf_importance || []);
        if (window.renderAeErrorChart) renderAeErrorChart(data.ae_recon_error || []);
      } catch (e) {
        this.explanation = { natural_language: 'Explain error: ' + e.message };
      }
    },

    consensusColorClass(val) {
      return { SAFE: 'tag-safe', CONFIRMED: 'tag-confirmed', KNOWN_PATTERN: 'tag-known_pattern', ZERO_DAY: 'tag-zero_day' }[val] || '';
    },

    // ── Batch CSV upload ──────────────────────────────────────────────────────

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
