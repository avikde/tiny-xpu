// ─── Dataset generation ────────────────────────────────────────────────────────
function genDataset(task, n = 200) {
  const xs = [], ys = [];
  const rng = seededRand(42);

  if (task === 'spiral') {
    for (let k = 0; k < 2; k++) {
      for (let i = 0; i < n / 2; i++) {
        const t = (i / (n / 2)) * 2 * Math.PI * 1.5;
        const r = 0.1 + 0.8 * (i / (n / 2));
        const nx = r * Math.cos(t + k * Math.PI) + rng() * 0.05;
        const ny = r * Math.sin(t + k * Math.PI) + rng() * 0.05;
        xs.push([nx, ny]);
        ys.push([k]);
      }
    }
  } else { // sine — 1D regression: y = sin(2πx) + sin(13 * x), normalized to [0,1]
    for (let i = 0; i < n; i++) {
      const x = rng() * 2 - 1;
      xs.push([x]);
      ys.push([(Math.sin(2 * Math.PI * x) + Math.sin(13 * x)) / 4 + 0.5]);
    }
  }
  return { xs, ys };
}

function seededRand(seed) {
  let s = seed;
  return function () {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

// ─── Neural net ───────────────────────────────────────────────────────────────
function makeNet(inDim, width, depth, outDim) {
  const layers = [];
  const sizes = [inDim, ...Array(depth).fill(width), outDim];
  const rng = seededRand(7);
  for (let i = 0; i + 1 < sizes.length; i++) {
    const fan_in = sizes[i], fan_out = sizes[i + 1];
    const scale = Math.sqrt(2 / fan_in);
    const W = Array.from({ length: fan_out }, () =>
      Array.from({ length: fan_in }, () => (rng() * 2 - 1) * scale)
    );
    const b = Array(fan_out).fill(0);
    layers.push({ W, b, fan_in, fan_out });
  }
  return layers;
}

function activation(x) { return Math.max(0, x); }
function actDeriv(x) { return x > 0 ? 1 : 0; }

function forward(layers, x) {
  const cache = [{ a: x, z: null }];
  let a = x;
  for (let li = 0; li < layers.length; li++) {
    const { W, b } = layers[li];
    const z = W.map((row, i) => row.reduce((s, w, j) => s + w * a[j], 0) + b[i]);
    const isLast = li === layers.length - 1;
    const aNext = isLast
      ? z.map(v => 1 / (1 + Math.exp(-v)))  // sigmoid output
      : z.map(v => activation(v));
    cache.push({ a: aNext, z });
    a = aNext;
  }
  return cache;
}

function loss(pred, target) {
  // binary cross-entropy (works for regression too since targets ∈ [0,1])
  const eps = 1e-7;
  return pred.reduce((s, p, i) => {
    const t = target[i];
    return s - (t * Math.log(p + eps) + (1 - t) * Math.log(1 - p + eps));
  }, 0) / pred.length;
}

function backward(layers, cache, target, lr = 0.05) {
  const L = layers.length;
  // output delta
  let delta = cache[L].a.map((p, i) => p - target[i]);

  for (let li = L - 1; li >= 0; li--) {
    const { W, b, fan_in } = layers[li];
    const aIn = cache[li].a;

    // weight + bias update
    for (let i = 0; i < delta.length; i++) {
      for (let j = 0; j < fan_in; j++) {
        W[i][j] -= lr * delta[i] * aIn[j];
      }
      b[i] -= lr * delta[i];
    }

    if (li === 0) break;

    // propagate delta
    const newDelta = Array(fan_in).fill(0);
    for (let j = 0; j < fan_in; j++) {
      for (let i = 0; i < delta.length; i++) {
        newDelta[j] += W[i][j] * delta[i];
      }
      newDelta[j] *= actDeriv(cache[li].z[j]);
    }
    delta = newDelta;
  }
}

function predict(layers, x) {
  const cache = forward(layers, x);
  return cache[cache.length - 1].a[0];
}

function countParams(depth, width) {
  const inDim = state.task === 'sine' ? 1 : 2;
  if (depth === 0) return width + 1;
  return (inDim * width + width) + (depth - 1) * (width * width + width) + (width + 1);
}
