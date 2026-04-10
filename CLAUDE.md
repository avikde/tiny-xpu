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
