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
      if (window.updateCharts) updateCharts(this.alerts);
    },

    async loadMetrics() {
      const res = await fetch('/metrics/');
      this.metrics = await res.json();
    },

    async selectAlert(alert) {
      this.selectedAlert = alert;
      // In a real scenario we'd have stored the original features; here we show a placeholder
      this.explanation = { natural_language: 'Explanation requires original flow features.' };
      if (window.renderShapChart) renderShapChart([]);
    },

    toggleReplay() {
      if (this.replaying) {
        this.stopReplay();
      } else {
        this.startReplay();
      }
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

    // ── Training controls ─────────────────────────────────────────────────────

    async startTraining() {
      if (this.training.running) return;
      this.training.running = true;
      this.training.currentRun = null;
      try {
        const res = await fetch('/training/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            stage: this.training.stage,
            data_source: this.training.dataSource,
          }),
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
      return {
        queued:  'tag-known_pattern',
        running: 'tag-zero_day',
        done:    'tag-safe',
        failed:  'tag-confirmed',
      }[status] || '';
    },
  };
}
