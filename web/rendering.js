// ─── ML Canvas ────────────────────────────────────────────────────────────────
const CANVAS_SIZE = 300;

function hiDPI(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || canvas.width;
  const cssH = canvas.clientHeight || canvas.height;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  canvas.style.width = cssW + 'px';
  canvas.style.height = cssH + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return ctx;
}

const GRID_STEP = 5;

function renderML() {
  const canvas = document.getElementById('mlCanvas');
  const ctx = hiDPI(canvas);
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  if (!currentNet) return;

  const targetFn = state.task === 'sine' ? sineTarget : squareTarget;
  render1D(ctx, targetFn);

  const params = countParams(state.depth, state.width);
  document.getElementById('statParams').textContent = params.toLocaleString();
  document.getElementById('statLoss').textContent = lastLoss != null ? Math.log10(lastLoss).toFixed(2) : '—';
}

function render1D(ctx, targetFn) {
  // Background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  const pad = 20;
  const W = CANVAS_SIZE - pad * 2, H = CANVAS_SIZE - pad * 2;
  const toX = x => pad + (x + 1) / 2 * W;
  const toY = y => pad + (1 - y) * H;

  // Grid lines at y=0.25, 0.5, 0.75
  ctx.strokeStyle = '#eaeef2';
  ctx.lineWidth = 1;
  [0.25, 0.5, 0.75].forEach(y => {
    ctx.beginPath(); ctx.moveTo(pad, toY(y)); ctx.lineTo(pad + W, toY(y)); ctx.stroke();
  });

  // Axes
  ctx.strokeStyle = '#d0d7de';
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, W, H);

  // True function
  ctx.strokeStyle = '#afb8c1';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let px = 0; px <= W; px++) {
    const x = (px / W) * 2 - 1;
    const y = targetFn(x);
    px === 0 ? ctx.moveTo(toX(x), toY(y)) : ctx.lineTo(toX(x), toY(y));
  }
  ctx.stroke();

  // Network prediction
  ctx.strokeStyle = '#0969da';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let px = 0; px <= W; px++) {
    const x = (px / W) * 2 - 1;
    const y = predict(currentNet, [x]);
    px === 0 ? ctx.moveTo(toX(x), toY(y)) : ctx.lineTo(toX(x), toY(y));
  }
  ctx.stroke();

  // Training points
  const { xs, ys } = dataset;
  for (let i = 0; i < xs.length; i++) {
    ctx.beginPath();
    ctx.arc(toX(xs[i][0]), toY(ys[i][0]), 2, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(101,109,118,0.35)';
    ctx.fill();
  }

  // Legend
  ctx.font = '10px -apple-system, sans-serif';
  ctx.fillStyle = '#afb8c1'; ctx.fillRect(pad + 4, pad + 4, 12, 3); ctx.fillStyle = '#656d76';
  ctx.fillText('true', pad + 20, pad + 10);
  ctx.fillStyle = '#0969da'; ctx.fillRect(pad + 4, pad + 16, 12, 3);
  ctx.fillStyle = '#656d76'; ctx.fillText('network', pad + 20, pad + 22);
}

// ─── HW Panel ────────────────────────────────────────────────────────────────
const BATCH = 1000;

const BW_BYTES_PER_CYCLE = 16; // SRAM bandwidth assumption

function hwMetrics() {
  const { width, depth, arrayRows, arrayCols } = state;
  const M = BATCH, N = width;
  const peakMacs = arrayRows * arrayCols;
  const spatialCols = Math.min(width, arrayCols) / arrayCols;  // N fills columns
  const spatialRows = Math.min(width, arrayRows) / arrayRows;  // K fills rows
  const spatial = spatialCols * spatialRows;
  const temporal = M / (M + arrayRows + N - 2);
  const overall = spatial * temporal;
  const throughput = overall * peakMacs;
  const latencyPerLayer = arrayRows + N - 2; // pipeline latency: first in → last out
  const totalLatency = depth * latencyPerLayer;
  // AI = useful MACs / total bytes (weights int8 + activations int8 + outputs int32)
  // Hidden layer: K=N=width, so weight_bytes=N², activation_bytes=M*N, output_bytes=4*M*N
  const ai = (M * N * N) / (N * N + M * N + 4 * M * N);
  return { spatial, temporal, overall, peakMacs, throughput, totalLatency, ai };
}

function barColor(v) {
  if (v >= 0.7) return '#1a7f37';
  if (v >= 0.4) return '#9a6700';
  return '#d1242f';
}

// Higher latency fraction = worse (inverted color scale)
function latencyBarColor(v) {
  if (v <= 0.3) return '#1a7f37';
  if (v <= 0.6) return '#9a6700';
  return '#d1242f';
}

function setBar(barId, valId, v) {
  const pct = (v * 100).toFixed(1);
  document.getElementById(barId).style.width = pct + '%';
  document.getElementById(barId).style.background = barColor(v);
  document.getElementById(valId).textContent = pct + '%';
}

function updateHW() {
  const m = hwMetrics();
  const { spatial, temporal, overall, peakMacs, throughput, totalLatency, ai } = m;
  setBar('hwUtilBar', 'hwUtil', overall);
  setBar('hwSpatialBar', 'hwSpatial', spatial);
  setBar('hwTemporalBar', 'hwTemporal', temporal);
  document.getElementById('hwThroughput').textContent = throughput.toFixed(1);
  const tpBar = document.getElementById('hwThroughputBar');
  tpBar.style.width = (throughput / peakMacs * 100).toFixed(1) + '%';
  tpBar.style.background = barColor(throughput / peakMacs);

  document.getElementById('hwLatency').textContent = totalLatency.toLocaleString();
  const maxLatency = 150;
  const latFrac = Math.min(totalLatency / maxLatency, 1);
  const latBar = document.getElementById('hwLatencyBar');
  latBar.style.width = (latFrac * 100).toFixed(1) + '%';
  latBar.style.background = latencyBarColor(latFrac);

  drawRoofline(peakMacs, throughput, ai);

  // Layer table
  const tbody = document.getElementById('layerTable');
  tbody.innerHTML = '';
  const { depth, width } = state;
  const shapes = [[1, width], ...Array(depth - 1).fill([width, width]), [width, 1]];
  shapes.forEach(([inn, out], i) => {
    const p = inn * out + out;
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>L${i + 1}</td><td>${inn}→${out}</td><td>${p.toLocaleString()}</td>`;
    tbody.appendChild(tr);
  });
}

function drawRoofline(peakMacs, throughput, ai) {
  const canvas = document.getElementById('rooflineCanvas');
  const cssW = 280, cssH = 160;
  const ctx = hiDPI(canvas);
  const W = cssW, H = cssH;
  ctx.clearRect(0, 0, W, H);

  const lPad = 42, rPad = 10, tPad = 12, bPad = 28;
  const plotW = W - lPad - rPad, plotH = H - tPad - bPad;

  const xMin = 0.4, xMax = 512;
  const yMin = 0.4, yMax = Math.max(peakMacs * 4, 64);

  const lx = v => lPad + Math.log(v / xMin) / Math.log(xMax / xMin) * plotW;
  const ly = v => tPad + (1 - Math.log(Math.max(v, yMin) / yMin) / Math.log(yMax / yMin)) * plotH;

  // Grid lines at powers of 2
  ctx.strokeStyle = '#eaeef2';
  ctx.lineWidth = 1;
  for (let v = 1; v <= xMax; v *= 2) {
    ctx.beginPath(); ctx.moveTo(lx(v), tPad); ctx.lineTo(lx(v), tPad + plotH); ctx.stroke();
  }
  for (let v = 1; v <= yMax; v *= 2) {
    ctx.beginPath(); ctx.moveTo(lPad, ly(v)); ctx.lineTo(lPad + plotW, ly(v)); ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = '#d0d7de';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(lPad, tPad); ctx.lineTo(lPad, tPad + plotH); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(lPad, tPad + plotH); ctx.lineTo(lPad + plotW, tPad + plotH); ctx.stroke();

  // Axis tick labels
  ctx.fillStyle = '#8c959f';
  ctx.font = '9px -apple-system, sans-serif';
  ctx.textAlign = 'center';
  for (let v = 1; v <= xMax; v *= 4) {
    ctx.fillText(v, lx(v), tPad + plotH + 10);
  }
  ctx.textAlign = 'right';
  for (let v = 1; v <= yMax; v *= 4) {
    if (ly(v) >= tPad) ctx.fillText(v, lPad - 3, ly(v) + 3);
  }

  // Axis labels
  ctx.fillStyle = '#656d76';
  ctx.font = '9px -apple-system, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Arithmetic Intensity (MACs/byte)', lPad + plotW / 2, H - 2);
  ctx.save();
  ctx.translate(9, tPad + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('MACs/cycle', 0, 0);
  ctx.restore();

  const BW_HI = 64; // high-bandwidth ceiling (matches plot_roofline.py bw_hi)
  const ridge = peakMacs / BW_BYTES_PER_CYCLE;
  const ridgeHi = peakMacs / BW_HI;

  // High-BW slope (solid, 64 B/cyc)
  ctx.strokeStyle = '#afb8c1';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(lx(xMin), ly(BW_HI * xMin));
  ctx.lineTo(lx(Math.min(ridgeHi, xMax)), ly(Math.min(BW_HI * ridgeHi, peakMacs)));
  ctx.stroke();

  // Low-BW slope (dashed, 16 B/cyc)
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(lx(xMin), ly(BW_BYTES_PER_CYCLE * xMin));
  ctx.lineTo(lx(Math.min(ridge, xMax)), ly(Math.min(BW_BYTES_PER_CYCLE * ridge, peakMacs)));
  ctx.stroke();
  ctx.setLineDash([]);

  // Compute ceiling (compute-bound region)
  ctx.strokeStyle = '#57606a';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(lx(Math.max(ridgeHi, xMin)), ly(peakMacs));
  ctx.lineTo(lx(xMax), ly(peakMacs));
  ctx.stroke();

  // Labels
  ctx.font = 'bold 9px -apple-system, sans-serif';
  ctx.fillStyle = '#57606a';
  ctx.textAlign = 'right';
  ctx.fillText(`Peak: ${peakMacs} MACs/cyc`, lPad + plotW - 2, ly(peakMacs) - 3);
  ctx.fillStyle = '#8c959f';
  ctx.font = '9px -apple-system, sans-serif';
  ctx.textAlign = 'left';
  if (ly(BW_HI * xMin) < tPad + plotH - 10) {
    ctx.fillText(`${BW_HI} B/cyc`, lx(xMin) + 2, ly(BW_HI * xMin) - 3);
  }
  if (ly(BW_BYTES_PER_CYCLE * xMin) < tPad + plotH - 10) {
    ctx.fillText(`${BW_BYTES_PER_CYCLE} B/cyc`, lx(xMin) + 2, ly(BW_BYTES_PER_CYCLE * xMin) - 3);
  }

  // Operating point
  const memBound = ai < ridge;
  const ptColor = memBound ? '#d1242f' : '#1a7f37';
  const px = lx(ai), py = ly(throughput);
  ctx.beginPath();
  ctx.arc(px, py, 5, 0, 2 * Math.PI);
  ctx.fillStyle = ptColor;
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Operating point label
  ctx.fillStyle = ptColor;
  ctx.font = 'bold 9px -apple-system, sans-serif';
  ctx.textAlign = py < tPad + 20 ? 'left' : 'center';
  ctx.fillText(memBound ? 'memory-bound' : 'compute-bound', px, py < tPad + 16 ? py + 14 : py - 8);
}

// ─── Insight callout ──────────────────────────────────────────────────────────
function updateInsight() {
  const { depth, width, arrayRows, arrayCols } = state;
  const { overall } = hwMetrics();
  const params = countParams(depth, width);
  const lossStr = lastLoss != null ? Math.log10(lastLoss).toFixed(2) : '?';

  const profile = depth >= 3 ? 'deep-narrow' : 'shallow-wide';
  const utilColor = overall >= 0.7 ? 'good' : overall >= 0.4 ? 'moderate' : 'poor';
  const utilLabel = { good: '✓ good', moderate: '~ moderate', poor: '✗ poor' }[utilColor];

  // Suggest twin
  let twinMsg = '';
  if (depth >= 3) {
    const twinDepth = 1;
    // find width with similar param count
    let bestW = 4, bestDiff = Infinity;
    for (const w of [4, 8, 12, 16, 24, 32]) {
      const diff = Math.abs(countParams(twinDepth, w) - params);
      if (diff < bestDiff) { bestDiff = diff; bestW = w; }
    }
    const twinSpatialCols = Math.min(bestW, arrayCols) / arrayCols;
    const twinSpatialRows = Math.min(bestW, arrayRows) / arrayRows;
    const twinUtil = twinSpatialCols * twinSpatialRows * (BATCH / (BATCH + arrayRows + bestW - 2));
    twinMsg = ` For similar capacity (~${countParams(twinDepth, bestW).toLocaleString()} params), a <strong>1-hidden-layer width-${bestW}</strong> network runs at <strong>${(twinUtil * 100).toFixed(0)}% utilization</strong> with only 1 sequential matmul — but may underfit.`;
  } else {
    const twinDepth = 5;
    let bestW = 4, bestDiff = Infinity;
    for (const w of [4, 8, 12, 16, 24, 32]) {
      const diff = Math.abs(countParams(twinDepth, w) - params);
      if (diff < bestDiff) { bestDiff = diff; bestW = w; }
    }
    const twinSpatialCols = Math.min(bestW, arrayCols) / arrayCols;
    const twinSpatialRows = Math.min(bestW, arrayRows) / arrayRows;
    const twinUtil = twinSpatialCols * twinSpatialRows * (BATCH / (BATCH + arrayRows + bestW - 2));
    twinMsg = ` For similar capacity (~${countParams(twinDepth, bestW).toLocaleString()} params), a <strong>5-hidden-layer width-${bestW}</strong> network runs at <strong>${(twinUtil * 100).toFixed(0)}% utilization</strong> with 5 sequential matmuls — but may be harder to train.`;
  }

  const { throughput, totalLatency, ai, peakMacs } = hwMetrics();
  const memBound = ai < peakMacs / BW_BYTES_PER_CYCLE;

  document.getElementById('insightBox').innerHTML =
    `This <strong>${profile}</strong> network (${params.toLocaleString()} params, ${depth} hidden layers) achieves log&#8321;&#8320; loss <strong>${lossStr}</strong>. ` +
    `On the ${arrayRows}×${arrayCols} array it runs at <strong>${throughput.toFixed(1)} MACs/cycle</strong> (${(overall * 100).toFixed(0)}% of peak ${peakMacs}) — ${utilLabel}, ` +
    `<strong>${memBound ? 'memory-bound' : 'compute-bound'}</strong> (AI = ${ai.toFixed(1)} MACs/byte). ` +
    `Pipeline latency: <strong>${totalLatency} cycles</strong> (${depth} layer${depth > 1 ? 's' : ''} × ${arrayRows + width - 2} cycles/layer). ` +
    twinMsg;
}

// ─── Network diagram ──────────────────────────────────────────────────────────
function drawNetDiagram() {
  const inDim = 1;
  const { depth, width } = state;
  const sizes = [inDim, ...Array(depth).fill(width), 1];
  const nLayers = sizes.length - 1;

  const BOX_W = 76, BOX_H = 42, ARROW = 20, END_W = 44, END_H = 28;
  const PAD = 16, SVG_H = 80;
  const cy = SVG_H / 2;
  const totalW = PAD + END_W + ARROW + nLayers * BOX_W + (nLayers - 1) * ARROW + ARROW + END_W + PAD;

  const parts = [];
  parts.push('<defs><marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">' +
    '<path d="M0,0 L0,6 L6,3 z" fill="#8c959f"/></marker></defs>');

  let x = PAD;
  parts.push(netPill(x, cy, END_W, END_H, `[${inDim}]`));
  x += END_W;

  for (let i = 0; i < nLayers; i++) {
    const isLast = i === nLayers - 1;
    parts.push(netArrow(x, cy, ARROW));
    x += ARROW;
    parts.push(netBox(x, cy, BOX_W, BOX_H, `${sizes[i]}→${sizes[i + 1]}`, isLast ? 'Linear' : 'ReLU', isLast));
    x += BOX_W;
  }

  parts.push(netArrow(x, cy, ARROW));
  x += ARROW;
  parts.push(netPill(x, cy, END_W, END_H, '[1]'));

  document.getElementById('netDiagram').innerHTML =
    `<svg xmlns="http://www.w3.org/2000/svg" width="${totalW}" height="${SVG_H}" style="display:block">${parts.join('')}</svg>`;
}

function netPill(x, cy, w, h, label) {
  const rx = h / 2;
  return `<rect x="${x}" y="${cy - h / 2}" width="${w}" height="${h}" rx="${rx}" fill="#f6f8fa" stroke="#8c959f" stroke-width="1.5"/>` +
    `<text x="${x + w / 2}" y="${cy + 4}" text-anchor="middle" font-size="11" font-family="-apple-system,sans-serif" fill="#57606a" font-weight="600">${label}</text>`;
}

function netBox(x, cy, w, h, line1, line2, isOutput) {
  const bg = isOutput ? '#f0fdf4' : '#ddf4ff';
  const border = isOutput ? '#1a7f37' : '#0969da';
  const color = isOutput ? '#1a7f37' : '#0969da';
  return `<rect x="${x}" y="${cy - h / 2}" width="${w}" height="${h}" rx="5" fill="${bg}" stroke="${border}" stroke-width="1.5"/>` +
    `<text x="${x + w / 2}" y="${cy - 3}" text-anchor="middle" font-size="10" font-family="-apple-system,sans-serif" fill="${color}" font-weight="600">${line1}</text>` +
    `<text x="${x + w / 2}" y="${cy + 10}" text-anchor="middle" font-size="9" font-family="-apple-system,sans-serif" fill="#656d76">${line2}</text>`;
}

function netArrow(x, cy, len) {
  return `<line x1="${x}" y1="${cy}" x2="${x + len - 1}" y2="${cy}" stroke="#8c959f" stroke-width="1.5" marker-end="url(#arr)"/>`;
}
