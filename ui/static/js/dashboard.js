// 42 required features (RF Gini importance > 1%)
const REQUIRED_FEATURES = [
  "Destination Port","Subflow Bwd Bytes","Init_Win_bytes_backward","Fwd Packet Length Max",
  "Flow IAT Mean","Subflow Bwd Packets","Min Packet Length","Flow IAT Std","Average Packet Size",
  "Flow IAT Max","Bwd Packet Length Mean","Bwd Packet Length Max","Subflow Fwd Bytes",
  "Total Length of Bwd Packets","Fwd Packet Length Std","Fwd IAT Mean","Bwd Packets/s",
  "Total Length of Fwd Packets","Fwd Header Length.1","Bwd Packet Length Min","Fwd IAT Total",
  "Flow Duration","Bwd URG Flags","Fwd URG Flags","ECE Flag Count","Down/Up Ratio","Fwd IAT Max",
  "Fwd Packet Length Mean","Fwd IAT Std","Bwd Header Length","Max Packet Length","Flow Bytes/s",
  "Packet Length Mean","Avg Fwd Segment Size","Subflow Fwd Packets","Bwd Packet Length Std",
  "Fwd IAT Min","Total Backward Packets","Total Fwd Packets","Flow IAT Min","Active Std",
  "Init_Win_bytes_forward"
];
// 28 optional features (Gini <= 1%, median-imputed if absent)
const OPTIONAL_FEATURES = [
  "Bwd IAT Mean","Flow Packets/s","Fwd Header Length","Avg Bwd Segment Size","Bwd IAT Total",
  "Bwd IAT Min","Idle Std","Bwd IAT Std","Active Max","Fwd Packets/s","SYN Flag Count",
  "Idle Mean","act_data_pkt_fwd","Bwd IAT Max","RST Flag Count","Fwd Packet Length Min",
  "PSH Flag Count","CWE Flag Count","Active Mean","Fwd PSH Flags","Packet Length Variance",
  "min_seg_size_forward","Active Min","Packet Length Std","ACK Flag Count","Bwd PSH Flags",
  "FIN Flag Count","URG Flag Count",
  "Idle Max","Idle Min"
];
const ALL_FEATURES = [...REQUIRED_FEATURES, ...OPTIONAL_FEATURES];

const CSV_HEADER = ALL_FEATURES.join(",");

// Sample rows: required columns only (optional columns are median-imputed by the server)
const BENIGN_ROW  = "443,13200,5,4,265,280,65,20,53.0,16.0,80,10,70.0,14.0,10500,4,1320.0,850.0,6000,100,5400,13200,0,0,0,1,4000,53.0,1400.0,180,90,20113.6,71.0,53.0,5,14.0,100,4,5,100,0.0,65536,65535,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
const ATTACK_ROW  = "22,100,500,0,0,0,0,0,0.0,0.0,0,0,0.0,0.0,0,0,0.2,0.0,1,0,50,100,0,0,0,0,1,0.0,0.0,0,0,0.0,0.0,0.0,500,0.0,0,0,500,0,0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";

function dashboard() {
  return {
    connected: false,
    replaying: false,
    alerts: [],
    metrics: {},
    selectedAlert: null,
    explanation: {},
    ws: null,
    activeTab: 'about',

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

    // Feature importance data for Models & Training tab
    featureGroups: { required: [], optional: [] },

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
      await this.loadMetrics();
      await this.loadModelVersions();
      await this.loadFeatureGroups();
      setInterval(() => this.loadMetrics(), 30000);
    },

    // Switch tab and lazy-load relevant data
    switchTab(tab) {
      this.activeTab = tab;
      if (tab === 'training') {
        this.loadTrainingHistory();
        this.loadModelVersions();
      } else if (tab === 'infer') {
        this.loadAlerts();
      }
    },

    // Load feature importance groups from model file
    async loadFeatureGroups() {
      try {
        const res = await fetch('/models/feature-importance');
        if (!res.ok) throw new Error('no endpoint');
        const data = await res.json();
        this.featureGroups = data;
      } catch (_) {
        // Fall back to static lists with uniform importance placeholder
        this.featureGroups = {
          required: REQUIRED_FEATURES.map((name, i) => ({ name, importance: 0.025 - i * 0.0005 })),
          optional: OPTIONAL_FEATURES.map((name, i) => ({ name, importance: 0.009 - i * 0.0003 })),
        };
      }
    },

    // Format batch 422 error (detail may be object or string)
    formatBatchError(err) {
      if (!err) return '';
      if (typeof err === 'string') return err;
      if (typeof err === 'object') {
        if (err.message) {
          const missing = err.missing_required_columns;
          if (missing?.length) return err.message + '\n\nMissing: ' + missing.join(', ');
          return err.message;
        }
        return JSON.stringify(err, null, 2);
      }
      return String(err);
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
        const res = await fetch(`/alerts/${alertId}/feedback`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ verdict }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        this.feedbackState[alertId] = verdict;
        await this.loadMetrics();   // refresh KPI strip + FAR/AMR immediately
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
