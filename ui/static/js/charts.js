const COLORS = {
  SAFE:          '#22c55e',
  KNOWN_PATTERN: '#f59e0b',
  CONFIRMED:     '#ef4444',
  ZERO_DAY:      '#a855f7',
};

let consensusChart = null;
let timelineChart  = null;
let shapChart      = null;

function buildConsensusChart(alerts) {
  const counts = { SAFE: 0, KNOWN_PATTERN: 0, CONFIRMED: 0, ZERO_DAY: 0 };
  alerts.forEach(a => { if (a.consensus in counts) counts[a.consensus]++; });

  const ctx = document.getElementById('consensusChart');
  if (!ctx) return;

  if (consensusChart) consensusChart.destroy();
  consensusChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: Object.keys(counts),
      datasets: [{
        data: Object.values(counts),
        backgroundColor: Object.keys(counts).map(k => COLORS[k]),
        borderWidth: 0,
      }],
    },
    options: {
      plugins: { legend: { labels: { color: '#e2e8f0', font: { size: 11 } } } },
      cutout: '65%',
    },
  });
}

function buildTimelineChart(alerts) {
  const last30 = [...alerts].reverse().slice(-30);
  const ctx = document.getElementById('timelineChart');
  if (!ctx) return;

  if (timelineChart) timelineChart.destroy();
  timelineChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: last30.map(a => new Date(a.timestamp).toLocaleTimeString()),
      datasets: [{
        label: 'AE Score',
        data: last30.map(a => a.ae_score ?? a.ae_score),
        borderColor: '#3b82f6',
        backgroundColor: '#3b82f620',
        fill: true,
        tension: 0.3,
        pointRadius: 2,
      }],
    },
    options: {
      scales: {
        x: { ticks: { color: '#64748b', maxTicksLimit: 6 }, grid: { color: '#2a2d3a' } },
        y: { ticks: { color: '#64748b' }, grid: { color: '#2a2d3a' } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function updateCharts(alerts) {
  buildConsensusChart(alerts);
  buildTimelineChart(alerts);
}

function pushTimelinePoint(row) {
  if (!timelineChart) return;
  const ds = timelineChart.data;
  ds.labels.push(new Date(row.timestamp).toLocaleTimeString());
  ds.datasets[0].data.push(row.ae_score);
  if (ds.labels.length > 60) {
    ds.labels.shift();
    ds.datasets[0].data.shift();
  }
  timelineChart.update('none');
}

function renderShapChart(shapValues) {
  const ctx = document.getElementById('shapChart');
  if (!ctx) return;
  if (shapChart) shapChart.destroy();
  if (!shapValues.length) return;

  shapChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: shapValues.map(d => d.feature),
      datasets: [{
        label: 'SHAP value',
        data: shapValues.map(d => d.shap_value),
        backgroundColor: shapValues.map(d => d.shap_value >= 0 ? '#ef4444' : '#3b82f6'),
      }],
    },
    options: {
      indexAxis: 'y',
      scales: {
        x: { ticks: { color: '#64748b' }, grid: { color: '#2a2d3a' } },
        y: { ticks: { color: '#e2e8f0', font: { size: 11 } } },
      },
      plugins: { legend: { display: false } },
    },
  });
}
