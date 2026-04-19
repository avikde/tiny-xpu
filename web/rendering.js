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
const BATCH = 64;

const BW_BYTES_PER_CYCLE = 16; // SRAM bandwidth assumption

// Hardware variants to compare
const HW_VARIANTS = [
  {
    name: 'TinyXPU',
    key: 'tinyxpu',
    color: '#0969da',      // Blue
    hasWeightLoading: false
  },
  {
    name: 'TPU-like',
    key: 'tpu',
    color: '#9a6700',      // Amber/Orange
    hasWeightLoading: true
  }
];

function hwMetricsAll() {
  const { width, depth, arrayRows, arrayCols, tinyxpuDoubleBuffer, tinyxpuOutputTaps } = state;
  const M = BATCH;
  const peakMacs = arrayRows * arrayCols;

  // Layer dimensions: [K, N] for each layer
  const layerDims = [];
  layerDims.push({ K: 1, N: width });  // Input layer: 1 → width
  for (let i = 1; i < depth; i++) {
    layerDims.push({ K: width, N: width });  // Hidden layers: width → width
  }
  layerDims.push({ K: width, N: 1 });  // Output layer: width → 1

  return HW_VARIANTS.map(variant => {
    let totalMacs = 0;
    let totalCycles = 0;
    let weightLoadCycles = 0;
    let computeCycles = 0;

    const isTinyXPU = variant.key === 'tinyxpu';

    layerDims.forEach(({ K, N }) => {
      // Calculate tiling dimensions
      const tilesK = Math.ceil(K / arrayRows);  // Tiles in K dimension (rows)
      const tilesN = Math.ceil(N / arrayCols);  // Tiles in N dimension (cols)
      const totalTiles = tilesK * tilesN;

      // MACs for this layer (same regardless of tiling)
      const macs = M * K * N;
      totalMacs += macs;

      // Determine tile dimensions (last tile may be smaller)
      const tileKEffective = Math.min(K, arrayRows);
      
      // Determine effective ROWS for each tile (output taps for TinyXPU)
      let effectiveRows = arrayRows;
      if (isTinyXPU && tinyxpuOutputTaps && tileKEffective < arrayRows) {
        const taps = [8, 12, 16].filter(t => t <= arrayRows);
        for (const tap of taps) {
          if (tap >= tileKEffective) {
            effectiveRows = tap;
            break;
          }
        }
      }

      // Calculate cycles per tile
      let cyclesPerTile = 0;
      let weightLoadPerTile = 0;

      if (isTinyXPU) {
        // TinyXPU with cascade-style weight loading
        if (tinyxpuDoubleBuffer) {
          // Full overlap: max(M, tileK) + effectiveRows per tile
          cyclesPerTile = Math.max(M, tileKEffective) + effectiveRows;
        } else {
          // Partial overlap: M + tileK + 1 + effectiveRows per tile
          cyclesPerTile = M + tileKEffective + 1 + effectiveRows;
        }
        weightLoadPerTile = tileKEffective;
      } else {
        // TPU-like: separate weight loading phase
        cyclesPerTile = M + effectiveRows + tileKEffective + effectiveRows;
        weightLoadPerTile = tileKEffective;
      }

      // Total cycles for all tiles (tiles run sequentially on same array)
      const layerCycles = cyclesPerTile * totalTiles;
      const layerWeightLoad = weightLoadPerTile * totalTiles;

      computeCycles += layerCycles - layerWeightLoad;
      weightLoadCycles += layerWeightLoad;
      totalCycles += layerCycles;
    });

    const throughput = totalMacs / totalCycles;  // MACs per cycle
    const overallUtil = throughput / peakMacs;

    // Arithmetic intensity (same for both variants)
    const totalBytes = layerDims.reduce((sum, { K, N }) => {
      return sum + (K * N) + (M * K) + (4 * M * N);  // weights + activations + outputs (int32)
    }, 0);
    const ai = totalMacs / totalBytes;

    return {
      name: variant.name,
      key: variant.key,
      color: variant.color,
      totalMacs,
      totalCycles,
      throughput,
      overallUtil,
      peakMacs,
      ai,
      // For stacked bar breakdown
      weightLoadCycles,
      computeCycles,
      weightLoadFraction: weightLoadCycles / totalCycles,
      computeFraction: computeCycles / totalCycles
    };
  });
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

function renderThroughputBars(metrics) {
  // Find max throughput for scaling
  const maxThroughput = Math.max(...metrics.map(m => m.throughput));

  const tinyxpu = metrics.find(m => m.key === 'tinyxpu');
  const tpu = metrics.find(m => m.key === 'tpu');

  // TinyXPU bar
  const tpTinyxpuPct = (tinyxpu.throughput / maxThroughput * 100).toFixed(1);
  document.getElementById('tpTinyxpuBar').style.width = tpTinyxpuPct + '%';
  document.getElementById('tpTinyxpuVal').textContent = tinyxpu.throughput.toFixed(1);

  // TPU-like bar
  const tpTpuPct = (tpu.throughput / maxThroughput * 100).toFixed(1);
  document.getElementById('tpTpuBar').style.width = tpTpuPct + '%';
  document.getElementById('tpTpuVal').textContent = tpu.throughput.toFixed(1);
}

function renderLatencyBars(metrics) {
  // Find max latency for scaling
  const maxLatency = Math.max(...metrics.map(m => m.totalCycles));

  const tinyxpu = metrics.find(m => m.key === 'tinyxpu');
  const tpu = metrics.find(m => m.key === 'tpu');

  // TinyXPU bar (single color - no weight loading)
  const latTinyxpuPct = (tinyxpu.totalCycles / maxLatency * 100).toFixed(1);
  document.getElementById('latTinyxpuBar').style.width = latTinyxpuPct + '%';
  document.getElementById('latTinyxpuVal').textContent = tinyxpu.totalCycles.toLocaleString();

  // TPU-like bar with breakdown (weight load + compute segments)
  const tpuWlPct = (tpu.weightLoadCycles / maxLatency * 100).toFixed(1);
  const tpuCompPct = (tpu.computeCycles / maxLatency * 100).toFixed(1);
  document.getElementById('latTpuWlBar').style.width = tpuWlPct + '%';
  document.getElementById('latTpuCompBar').style.width = tpuCompPct + '%';
  document.getElementById('latTpuVal').textContent = tpu.totalCycles.toLocaleString();
}

function updateHW() {
  const metrics = hwMetricsAll();

  renderThroughputBars(metrics);
  renderLatencyBars(metrics);

  drawRoofline(metrics);

  // Layer table with tiling info
  const tbody = document.getElementById('layerTable');
  tbody.innerHTML = '';
  const { depth, width, arrayRows, arrayCols } = state;
  const shapes = [[1, width], ...Array(depth - 1).fill([width, width]), [width, 1]];
  
  shapes.forEach(([inn, out], i) => {
    const p = inn * out + out;
    const K = inn;
    const N = out;
    
    // Calculate tiling
    const tilesK = Math.ceil(K / arrayRows);
    const tilesN = Math.ceil(N / arrayCols);
    const totalTiles = tilesK * tilesN;
    
    // Build tiling annotation
    let tilingInfo = '';
    if (totalTiles > 1) {
      tilingInfo = ` <span style="color:#9a6700; font-size:0.7em;">(${tilesK}×${tilesN} tiles)</span>`;
    }
    
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>L${i + 1}</td><td>${inn}→${out}${tilingInfo}</td><td>${p.toLocaleString()}</td>`;
    tbody.appendChild(tr);
  });
}

function drawRoofline(allMetrics) {
  const canvas = document.getElementById('rooflineCanvas');
  const cssW = 280, cssH = 160;
  const ctx = hiDPI(canvas);
  const W = cssW, H = cssH;
  ctx.clearRect(0, 0, W, H);

  const lPad = 42, rPad = 10, tPad = 12, bPad = 28;
  const plotW = W - lPad - rPad, plotH = H - tPad - bPad;

  // Use peakMacs from first metric (same for all variants)
  const peakMacs = allMetrics[0].peakMacs;

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

  const BW_HI = 64; // high-bandwidth ceiling
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

  // Draw operating points for each hardware variant
  allMetrics.forEach((m, idx) => {
    const px = lx(m.ai);
    const py = ly(m.throughput);

    // Draw dot with variant color (smaller radius = 3 to avoid overlap)
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, 2 * Math.PI);
    ctx.fillStyle = m.color;
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Label with variant name (offset to avoid overlap)
    ctx.fillStyle = m.color;
    ctx.font = 'bold 9px -apple-system, sans-serif';
    ctx.textAlign = 'left';
    const labelOffset = idx === 0 ? -10 : 8;
    ctx.fillText(m.name, px + 6, py + labelOffset);
  });
}

// ─── Insight callout ──────────────────────────────────────────────────────────
function updateInsight() {
  const { depth, width, arrayRows, arrayCols } = state;
  const metrics = hwMetricsAll();
  const params = countParams(depth, width);
  const lossStr = lastLoss != null ? Math.log10(lastLoss).toFixed(2) : '?';

  // Find TPU-like vs TinyXPU comparison
  const tinyxpu = metrics.find(m => m.key === 'tinyxpu');
  const tpu = metrics.find(m => m.key === 'tpu');
  const slowdown = ((tpu.totalCycles / tinyxpu.totalCycles - 1) * 100).toFixed(0);

  const memBound = tinyxpu.ai < tinyxpu.peakMacs / BW_BYTES_PER_CYCLE;

  // Check if tiling is needed for hidden layers
  const needsTiling = width > arrayRows || width > arrayCols;
  let tilingMsg = '';
  if (needsTiling) {
    const tilesK = Math.ceil(width / arrayRows);
    const tilesN = Math.ceil(width / arrayCols);
    tilingMsg = ` <strong style="color:#d1242f;">Requires ${tilesK}×${tilesN} tiling</strong> (width > array). `;
  }

  document.getElementById('insightBox').innerHTML =
    `This <strong>${depth}-layer width-${width}</strong> network (${params.toLocaleString()} params) achieves log&#8321;&#8320; loss <strong>${lossStr}</strong>. ` +
    `On the ${arrayRows}×${arrayCols} array: ` +
    `<span style="color:${tinyxpu.color}"><strong>TinyXPU</strong>: ${tinyxpu.throughput.toFixed(1)} MACs/cyc, ${tinyxpu.totalCycles.toLocaleString()} cycles</span> vs ` +
    `<span style="color:${tpu.color}"><strong>TPU-like</strong>: ${tpu.throughput.toFixed(1)} MACs/cyc, ${tpu.totalCycles.toLocaleString()} cycles</span>. ` +
    tilingMsg +
    `Weight loading adds <strong>${slowdown}%</strong> latency overhead. ` +
    `<strong>${memBound ? 'Memory-bound' : 'Compute-bound'}</strong> (AI = ${tinyxpu.ai.toFixed(1)} MACs/byte).`;
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
