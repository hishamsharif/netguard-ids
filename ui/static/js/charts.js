const COLORS = {
  SAFE:          '#22c55e',
  KNOWN_PATTERN: '#f59e0b',
  CONFIRMED:     '#ef4444',
  ZERO_DAY:      '#a855f7',
};

let consensusChart = null;
let timelineChart  = null;
let importanceChart = null;
let aeErrorChart    = null;

function buildConsensusChart(alerts) {
  const counts = { SAFE: 0, KNOWN_PATTERN: 0, CONFIRMED: 0, ZERO_DAY: 0 };
  alerts.forEach(a => { if (a.consensus in counts) counts[a.consensus]++; });

  const ctx = document.getElementById('consensusChart');
  if (!ctx) return;

  if (consensusChart) consensusChart.destroy();
  consensusChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Safe', 'Known Pattern', 'Confirmed Attack', 'Zero-Day'],
      datasets: [{
        data: Object.values(counts),
        backgroundColor: Object.keys(counts).map(k => COLORS[k]),
        borderWidth: 0,
      }],
    },
    options: {
      plugins: {
        legend: { labels: { color: '#e2e8f0', font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.parsed} flow${ctx.parsed !== 1 ? 's' : ''}`,
          },
        },
      },
      cutout: '65%',
    },
  });
}

function buildTimelineChart(alerts, threshold) {
  const last30 = [...alerts].reverse().slice(-30);
  const ctx = document.getElementById('timelineChart');
  if (!ctx) return;

  if (timelineChart) timelineChart.destroy();

  const thresholdVal = threshold ?? 0.008927;
  timelineChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: last30.map(a => new Date(a.timestamp).toLocaleTimeString()),
      datasets: [
        {
          label: 'AE Reconstruction Error',
          data: last30.map(a => a.ae_score),
          borderColor: '#3b82f6',
          backgroundColor: '#3b82f620',
          fill: true,
          tension: 0.3,
          pointRadius: 3,
          pointBackgroundColor: last30.map(a =>
            a.ae_score > thresholdVal ? '#ef4444' : '#22c55e'
          ),
        },
        {
          label: `Anomaly Threshold (${thresholdVal.toFixed(4)})`,
          data: last30.map(() => thresholdVal),
          borderColor: '#f59e0b',
          borderDash: [6, 3],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      scales: {
        x: { ticks: { color: '#64748b', maxTicksLimit: 6 }, grid: { color: '#2a2d3a' } },
        y: { ticks: { color: '#64748b' }, grid: { color: '#2a2d3a' } },
      },
      plugins: {
        legend: {
          display: true,
          labels: { color: '#94a3b8', font: { size: 11 }, boxWidth: 20 },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              if (ctx.datasetIndex === 1) return ` Threshold: ${thresholdVal.toFixed(6)}`;
              return ` AE Score: ${ctx.parsed.y.toFixed(6)}`;
            },
          },
        },
      },
    },
  });
}

function updateCharts(alerts, threshold) {
  buildConsensusChart(alerts);
  buildTimelineChart(alerts, threshold);
}

function pushTimelinePoint(row) {
  if (!timelineChart) return;
  const ds = timelineChart.data;
  ds.labels.push(new Date(row.timestamp).toLocaleTimeString());
  ds.datasets[0].data.push(row.ae_score);
  const thresh = timelineChart.data.datasets[1]?.data[0] ?? 0.008927;
  ds.datasets[0].pointBackgroundColor.push(row.ae_score > thresh ? '#ef4444' : '#22c55e');
  if (ds.labels.length > 60) {
    ds.labels.shift();
    ds.datasets[0].data.shift();
    ds.datasets[0].pointBackgroundColor.shift();
    ds.datasets[1].data.shift();
  }
  timelineChart.update('none');
}

function renderImportanceChart(rfImportance) {
  const ctx = document.getElementById('importanceChart');
  if (!ctx || !rfImportance?.length) return;
  if (importanceChart) importanceChart.destroy();

  importanceChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: rfImportance.map(d => d.feature),
      datasets: [{
        label: 'Gini Importance',
        data: rfImportance.map(d => d.importance),
        backgroundColor: '#3b82f6cc',
        borderRadius: 3,
      }],
    },
    options: {
      indexAxis: 'y',
      scales: {
        x: { ticks: { color: '#64748b' }, grid: { color: '#2a2d3a' } },
        y: { ticks: { color: '#e2e8f0', font: { size: 11 } } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => ` Importance: ${c.parsed.x.toFixed(4)}` } },
      },
    },
  });
}

function renderAeErrorChart(aeErrors) {
  const ctx = document.getElementById('aeErrorChart');
  if (!ctx || !aeErrors?.length) return;
  if (aeErrorChart) aeErrorChart.destroy();

  aeErrorChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: aeErrors.map(d => d.feature),
      datasets: [{
        label: 'Reconstruction Error',
        data: aeErrors.map(d => d.error),
        backgroundColor: '#a855f7cc',
        borderRadius: 3,
      }],
    },
    options: {
      indexAxis: 'y',
      scales: {
        x: { ticks: { color: '#64748b' }, grid: { color: '#2a2d3a' } },
        y: { ticks: { color: '#e2e8f0', font: { size: 11 } } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => ` Error: ${c.parsed.x.toFixed(6)}` } },
      },
    },
  });
}

// Legacy alias kept for any inline calls
function renderShapChart(data) { renderImportanceChart(data); }
