# CLAUDE.md

## Project Overview

SystemVerilog systolic array (weight-stationary) with a C++ ONNX Execution Provider that dispatches `MatMulInteger` to a Verilator simulation of the array.

**Languages:** SystemVerilog, C++17, Python  
**Key tools:** Verilator (SV→C++), Icarus Verilog (simulation), cocotb (Python-based SV testing), ONNX Runtime (plugin EP)

## Key Files

| Path | Purpose |
|------|---------|
| `src/pe.sv` | Processing Element — int8×int8→int32 MAC, weight-stationary |
| `src/array.sv` | ROWS×COLS PE grid with input skewing and output de-skewing |
| `onnx-plugin/src/tinyxpu_ep.cpp` | ONNX EP — receives `MatMulInteger`, drives the Verilator model |
| `onnx-plugin/src/tinyxpu_perf.cpp` | Performance counters |
| `test/test_pe.py` | cocotb tests for PE |
| `scripts/matmul.py` | Generates `matmul_integer_?x?.onnx` test models |
| `scripts/run_matmul.py` | End-to-end test: ONNX model → TinyXPU EP → verify vs NumPy |

## Python Environment

All Python scripts require the `.venv` virtualenv. Always activate it before running scripts or tests:

```sh
source .venv/bin/activate
python scripts/matmul.py   # or any other script
```

**For Claude:** When searching for files or code, skip `.venv/`, `test/sim_build/`, `test/sim_build_array/`, and `build/` — these are generated/dependency directories. Always run Python via the activated venv, not the system `python3.13` directly.

## Build

```sh
mkdir -p build && cd build
cmake .. -DSIM=ON   # -DSIM=ON links Verilator into the EP
make -j
```

`-DSIM=ON` is required to run with Verilator. `SIM_ROWS`/`SIM_COLS` (default 16) set the array size passed to Verilator via `-GROWS`/`-GCOLS`.

## Test

```sh
cd build && ctest --verbose
# Tests run test/test_*.py via cocotb + Icarus Verilog
# Waveforms written to test/sim_build/*.fst
```

## Architecture Notes

- **PE:** `weight_ld=1` latches weight; `en=1` streams data east and accumulates partial sums south. `int8×int8→int32` MAC.
- **Array:** Input skewing staggers row `k` by `k` cycles. Output de-skewing aligns all columns to the same valid tick. This gives near-100% MAC utilization as M grows.
- **EP flow:** `MatMulInteger(X[M,K], W[K,N])` → tile if needed → load weights (`weight_ld`) → stream activations (`en`) → collect `acc_out`.
- ONNX Runtime installed at `/opt/onnxruntime`.

## Web Tool (`web/`)

Single-file `web/index.html` — no build step, no dependencies. `web/serve.js` is a minimal Node.js static server for local testing.

**Neural net:** vanilla JS backprop, SGD lr=0.01, 500 steps, Xavier init. Tasks use a fixed seed so results are reproducible. Sine task uses 1D input `[x]`; spiral uses 2D `[x, y]` — `makeNet` and `countParams` branch on `state.task` accordingly.

**Rendering:** Spiral decision boundary samples a 300×300 canvas at 5px grid (3600 forward passes). Sine renders a 1D line plot (300 forward passes) — much cheaper. The roofline canvas is a log-log plot drawn with the Canvas 2D API; it re-renders on every control change without retraining.

**HW metrics** are pure arithmetic — no simulation. Bandwidth assumption is hardcoded at `BW_BYTES_PER_CYCLE = 16` B/cycle (matches `scripts/plot_roofline.py`). Batch size is hardcoded at `BATCH = 64`.

**Deployment:** `.github/workflows/pages.yml` deploys `web/` to GitHub Pages on push to `main`. The Pages environment is restricted to `main` only — do not add other branches to the workflow trigger.
