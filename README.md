# tiny-xpu

## Project goal

While there are other projects building up small (~2x2) TPU-inspired designs (see related projects below), this project has a salient combination of goals:

- Modular SystemVerilog setup to support non-rectangular systolic architectures
- Easy software interface via ONNX EP and maybe others
- Support for FPGA deployment

## Setup, build, and test

Set up in WSL or other Linux: 

- `sudo apt install iverilog` -- Icarus Verilog for simulation
- Install the [Surfer waveform viewer](https://marketplace.visualstudio.com/items?itemName=surfer-project.surfer) VSCode extension for viewing `.vcd` waveform files
- `sudo apt install yosys` -- Yosys for synthesis (or [build from source](https://github.com/YosysHQ/yosys) for the latest version)
- `sudo apt install verilator` -- Compile SV -> C++ for EP linkage

Set up a venv for python packages

```sh
python3 -m venv .venv
source .venv/bin/activate
# Python tool for more powerful SystemVerilog testing
pip install cocotb
# Run ONNX models
pip install onnxruntime==1.23.2 onnx
```

Build:

```shell
mkdir -p build && cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DSIM=ON
make -j
```

Important flags:
- `-DSIM=ON` - link Verilator into the ONNX EP so that it executes verilator simulation. When off, it will attempt to use hardware when implemented.

Test (optional):

```shell
cd build && ctest --verbose
```

Tests produce waveform files (`*.fst`) in `test/sim_build/`. Open them in VSCode with the Surfer extension to inspect signals.

## Running ONNX models

### 1. Build onnxruntime from source (Linux / WSL)

```bash
git clone --recursive https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
git checkout v1.23.2
# Build, skipping tests
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF
cmake --install build/Linux/RelWithDebInfo --prefix $HOME/onnxruntime_install
```

### 2. Configure Library Path

Add the ONNX Runtime library to your library path:

```bash
echo 'export LD_LIBRARY_PATH=$HOME/onnxruntime_install/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. Install ONNX

```sh
pip install onnx
```

### 4. Run the matmul ONNX model with tiny-xpu

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

Kung paper
- Network of PE's

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

### Networked PE's -> Systolic array

- Multiple PE's connected together form a systolic array

## Related projects

There are a number of "tiny TPU"-type projects, due to the current popularity of TPUs and LLMs.

- [tiny-tpu-v2/tiny-tpu](https://github.com/tiny-tpu-v2/tiny-tpu/tree/main) - 2x2 matmul + ReLU to solve XOR problem
- [Alanma23/tinytinyTPU](https://github.com/Alanma23/tinytinyTPU) - 2x2 matmul + ReLU / ReLU6

