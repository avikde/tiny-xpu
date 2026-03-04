# tiny-xpu

## Project goal

While there are other projects building up small (~2x2) TPU-inspired designs (see related projects below), this project has a salient combination of goals:

- Modular SystemVerilog setup to support non-rectangular systolic architectures
- Easy software interface via ONNX EP and maybe others
- Scaffolding to evaluate architectural tradeoffs, include performance counters
- Support for FPGA deployment

## Setup, build, and test

Set up in WSL or other Linux: 

- `sudo apt install iverilog` -- Icarus Verilog for simulation
- `sudo apt install yosys` -- Yosys for synthesis (or [build from source](https://github.com/YosysHQ/yosys) for the latest version)
- `sudo apt install verilator` -- Compile SV -> C++ for EP linkage
- Install pre-built onnxruntime (check https://github.com/microsoft/onnxruntime/releases) -- this is used to build the ONNX EP C++ library

```bash
sudo mkdir -p /opt/onnxruntime
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-1.24.2.tgz
sudo tar -xzf onnxruntime-linux-x64-1.24.2.tgz -C /opt/onnxruntime --strip-components=1
```

- Add the ONNX Runtime library to your library path:

```bash
echo 'export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Set up a venv for python packages

```sh
python3 -m venv .venv
source .venv/bin/activate
# Python tool for more powerful SystemVerilog testing
pip install cocotb
# Run ONNX models (matching onnxruntime version to the downloaded release)
pip install onnxruntime==1.24.2 onnx
```

Build:

```shell
mkdir -p build && cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DSIM=ON
make -j
```

Important flags:
- `-DSIM=ON` - link Verilator into the ONNX EP so that it executes verilator simulation. When off, it will attempt to use hardware when implemented.

### Test and view waveforms (optional)

- Install the [Surfer waveform viewer](https://marketplace.visualstudio.com/items?itemName=surfer-project.surfer) VSCode extension for viewing `.vcd` waveform files

```shell
cd build && ctest --verbose
```

Tests produce waveform files (`*.fst`) in `test/sim_build/`. Open them in VSCode with the Surfer extension to inspect signals.

## Run the matmul ONNX model with tiny-xpu

The end-to-end flow is: generate an ONNX model → run it through `onnxruntime`
with the TinyXPU execution provider, which dispatches `MatMulInteger` to the
Verilator simulation of the systolic array.

**Step 1 — build** (see above, must use `-DSIM=ON`).

**Step 2 — generate the ONNX model:**

```shell
cd scripts
python3 matmul.py
# writes matmul_integer_4x4.onnx
```

The model contains a single `MatMulInteger` node:
- `X (int8, [M, 4])`
- `W (int8, [4, 4])`
- `Y (int32, [M, 4]) = MatMulInteger(X, W)`

Constraints: `K = N = 4` (the hardware array dimensions).
`M` is unrestricted — the array streams one row of `X` at a time.

**Step 3 — run with the TinyXPU EP:**

```shell
python3 run_matmul.py
```

The script registers the plugin EP, loads the model, feeds a 4×4 `int8`
input, verifies the `int32` result against a NumPy reference, and prints
`PASS` on success.  It replaces the old `onnx-plugin/test/test_tinyxpu_ep.py`
development harness.

## Systolic array implementation

The systolic array is a `ROWS × COLS` grid of processing elements (PEs) connected in a mesh. Each PE performs one multiply-accumulate per cycle. The array size is set by `ROWS` and `COLS` parameters, overridable at elaboration time (e.g. via Verilator's `-GROWS=N -GCOLS=N`).

Current dataflow is **weight-stationary**: weights are loaded once into the PE grid, then activations stream east (→) through each row while partial sums cascade south (↓) through each column, accumulating as they go. This maximises weight reuse — each weight participates in every row of the output tile without being reloaded.

Input and output ports of `array.sv`:
- `data_in[ROWS]` — one int8 activation per row, presented sequentially (one output row per streaming cycle); internal skew registers stagger them automatically
- `weight_in[ROWS*COLS]`, `weight_ld` — load all PE weights in one cycle before streaming begins
- `acc_out[COLS]` — one int32 result per column, de-skewed so all columns are valid at the same cycle

> **Note on configurable dataflow:** It is feasible to add a `DATAFLOW` parameter (weight-stationary vs output-stationary) switchable via a CMake option that passes `-GDATAFLOW=0/1` to Verilator. However, the two modes differ enough in their weight-loading interface (`weight_in[ROWS*COLS]` broadcast vs `weight_in[COLS]` streaming) that a clean unified port is awkward in SV. The most practical approach would be separate `array_ws.sv` / `array_os.sv` files, selected by the CMake `ARRAY_DATAFLOW` option, sharing a common `pe.sv` modified with a `generate if` for the accumulation logic.

### PE (`pe.sv`)

Processing Element (PE) for systolic array, named as in Kung (1982)

- Performs multiply-accumulate: `acc += weight * data_in`
- Passes data through to neighboring PEs via `data_out`
- The PE does `int8 × int8 → int32`, then `int32 + int32 → int32`
- `int8×int8→int32` is the standard choice (used by [Google's TPUs](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e), [Arm NEON `sdot`](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdot_s32), etc.)

In a systolic array, there are two distinct phases:

1. Weight loading phase (`weight_ld=1, en=0`): Before computation begins, you load each PE with its weight from the weight matrix. In a 2x2 systolic array doing `C = A × B`, each PE gets one element of B. This happens once per matrix multiply (or once per tile, for larger matrices).
2. Compute phase (`weight_ld=0, en=1`): The weights stay "stationary" (this is the weight-stationary dataflow). Input activations stream through via data_in/data_out, and partial sums accumulate via acc_in/acc_out. The weights don't change during this phase.

So the typical sequence is:

- Load weights for all PEs (a few cycles with `weight_ld=1`)
- Stream many inputs through with weights held fixed (`en=1, weight_ld=0`)
- When you need new weights (next layer, next tile), load again

Data flows east (→), partial sums flow south (↓) — this is the standard output-stationary / weight-stationary systolic layout from Kung (1982).

```
        weight_ld
            │
            │  en
            ▼  ▼
         ┌──────────┐   
         │    PE    ├──► data_out
         │          │
         │  weight  │
         │  (reg)   │
data_in─►│          │
         │  ×  +    │
acc_in ──►          ├──► acc_out
         └──────────┘
```

### Input skewing

The standard way to fully utilize a weight-stationary systolic array is **input skewing**: PE row `k` receives its activation one cycle later than PE row `k-1`. For a 4×4 array computing `C = A × B` with M output rows, the driver presents:

```
Cycle:   0      1      2      3      4     ...   M+2
Row 0: a00    a10    a20    a30    ...
Row 1:  0     a01    a11    a21    a31    ...
Row 2:  0      0     a02    a12    a22    ...
Row 3:  0      0      0     a03    a13    ...
```

With skewing, M output rows flow through the pipeline in `M + (ROWS+COLS−1)` total streaming ticks instead of `M × (ROWS+COLS)`, so MAC efficiency approaches 100% as M grows (weight reuse AND compute utilisation both improve). Without skewing the current driver pays the full pipeline fill/drain cost per row, capping MAC efficiency at 12.5% regardless of M.

The skewed input stream must be de-skewed on the output side: `acc_out[j]` for output row `i` is valid at tick `i + ROWS + j`, not all at the same tick.

## Related projects

There are a number of "tiny TPU"-type projects, due to the current popularity of TPUs and LLMs.

- [tiny-tpu-v2/tiny-tpu](https://github.com/tiny-tpu-v2/tiny-tpu/tree/main) - 2x2 matmul + ReLU to solve XOR problem
- [Alanma23/tinytinyTPU](https://github.com/Alanma23/tinytinyTPU) - 2x2 matmul + ReLU / ReLU6
