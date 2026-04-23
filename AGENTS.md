# AGENTS.md

## Project

SystemVerilog weight-stationary systolic array with a C++ ONNX Runtime Execution Provider. The EP dispatches `MatMulInteger` to a Verilator simulation of the array.

**Languages:** SystemVerilog, C++17, Python  
**Tools:** Verilator (SV→C++), Icarus Verilog (cocotb tests), ONNX Runtime (plugin EP), cmake, cocotb

## Python Environment

All Python work requires the `.venv` virtualenv. **Always activate it** before running scripts or tests.

```sh
source .venv/bin/activate
python scripts/matmul.py
```

Do **not** use the system `python3.13` directly.

## Build

```sh
mkdir -p build && cd build
cmake .. -DSIM=ON   # required to link Verilator into the EP
make -j
```

Key CMake flags:
- `-DSIM=ON` — enables Verilator backend (required for end-to-end runs)
- `-DSIM_ROWS=N -DSIM_COLS=N` — override array size (default **64×64**)

The EP shared library is built at `build/onnx-plugin/libtinyxpu_ep.{so,dylib}`.

ONNX Runtime is expected at `/opt/onnxruntime`. On macOS, ensure `DYLD_LIBRARY_PATH` includes `/opt/onnxruntime/lib`.

## Test

```sh
cd build && ctest --verbose
```

- Tests are `test/test_*.py` run via cocotb + Icarus Verilog
- **cocotb exits 0 even on test failures** — each test script parses `results.xml` and exits 1 manually
- `test/test_pe.py` simulates the `pe` module; `test/test_array.py` simulates `array` at **16×16** (hardcoded in the test, passed via `-Parray.ROWS=16 -Parray.COLS=16`)
- `test_array.py` sets `always=True` in the runner to avoid stale binaries from the `pe` test
- Waveforms: `test/sim_build/*.fst` (PE) and `test/sim_build_array/*.fst` (array)

## Running End-to-End

```sh
source .venv/bin/activate
python scripts/matmul.py          # generates .onnx models (including 16x16)
python scripts/run_matmul.py      # ONNX → TinyXPU EP → verify vs NumPy
```

`run_matmul.py` defaults to `scripts/matmul_integer_16x16.onnx` and the plugin at `build/onnx-plugin/libtinyxpu_ep.{so,dylib}`. It will fail with a helpful message if either is missing.

## Architecture Notes

- **PE (`pe.sv`):** `weight_ld=1` latches weight via `acc_in` (dual-use port); `en=1` streams `data_in` east and accumulates partial sums south. `int8×int8→int32` MAC.
- **Array (`array.sv`):** No internal de-skewing or input delay lines. The caller is responsible for externally pre-staggering inputs: row `r` must begin `r` cycles after row 0. This enables direct layer chaining without inter-layer buffers. Output `acc_out[c]` for row `i` is valid at cycle `i + ROWS + c`.

## Search / Ignore Rules

Skip these generated directories when searching or grepping:
- `.venv/`
- `build/`
- `test/sim_build/`
- `test/sim_build_array/`

## Web Tool (`web/`)

- Single-file `web/index.html` — no build step, no dependencies
- Local server: `node web/serve.js` (serves at `localhost:3000`)
- Deployed to GitHub Pages by `.github/workflows/pages.yml` on pushes to `main` only

## Common Mistakes to Avoid

- Forgetting `source .venv/bin/activate` before Python scripts or tests
- Building without `-DSIM=ON` and then trying to run `run_matmul.py`
- Changing `array.sv` ROWS/COLS parameters in source — use cmake `-DSIM_ROWS`/`-DSIM_COLS` instead for Verilator builds
- Trusting cocotb's exit code alone — it returns 0 on test failures; rely on the `results.xml` parsing in the test scripts
