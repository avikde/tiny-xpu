# tiny-xpu

A modular SystemVerilog systolic array with an ONNX Execution Provider, targeting FPGA deployment and architectural exploration.

**Goals:** Non-rectangular systolic architectures · ONNX EP software interface · Performance counters for architectural tradeoff analysis · FPGA deployment support

## Setup

### Linux (Debian/Ubuntu)

```sh
sudo apt install iverilog verilator yosys
```

Install ONNX Runtime:
```bash
sudo mkdir -p /opt/onnxruntime
wget -P /tmp https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-1.24.2.tgz
sudo tar -xzf /tmp/onnxruntime-linux-x64-1.24.2.tgz -C /opt/onnxruntime --strip-components=1
echo 'export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH' >> ~/.bashrc && source ~/.bashrc
```

### Mac (Homebrew)

```sh
brew install iverilog verilator yosys cmake
brew install python@3.13  # open new terminal after
```

Install ONNX Runtime:
```bash
sudo mkdir -p /opt/onnxruntime
curl -OL --output-dir /tmp https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-osx-arm64-1.24.2.tgz
sudo tar -xzf /tmp/onnxruntime-osx-arm64-1.24.2.tgz -C /opt/onnxruntime --strip-components=1
echo 'export DYLD_LIBRARY_PATH=/opt/onnxruntime/lib:$DYLD_LIBRARY_PATH' >> ~/.zshrc && source ~/.zshrc
```

### Python packages

```sh
python3.13 -m venv .venv && source .venv/bin/activate
pip install matplotlib cocotb onnxruntime==1.24.2 onnx
```

`optimum` brings in `transformers` and `huggingface_hub` and is the standard tool for exporting and quantizing HuggingFace models to ONNX.

## Build and Test

```sh
mkdir -p build && cd build
cmake .. -DSIM=ON   # SIM=ON links Verilator into the ONNX EP
make -j
ctest --verbose     # waveforms written to test/sim_build/*.fst
```

Key CMake flags:
- `-DSIM=ON` — use Verilator simulation backend (required for software runs)
- `-DSIM_ROWS=N -DSIM_COLS=N` — override array size (default 16×16)

Install the [Surfer](https://marketplace.visualstudio.com/items?itemName=surfer-project.surfer) VSCode extension to view `.fst` waveforms.

## Run scripts

```sh
source .venv/bin/activate

python scripts/matmul.py          # generates matmul_integer_?x?.onnx
python scripts/run_matmul.py      # 2-D MatMulInteger via Verilator, verifies vs NumPy
python scripts/test_ops.py        # batched MatMulInteger + Gemm tests
```

## Systolic Array Architecture

A `ROWS × COLS` PE grid. Dataflow is **weight-stationary**: weights load once, then activations stream east (→) while partial sums cascade south (↓).

**Ports of `array.sv`:**
- `data_in[ROWS]` — one int8 activation per row per cycle (internally skewed)
- `weight_in[ROWS*COLS]`, `weight_ld` — load all weights in one cycle
- `acc_out[COLS]` — one int32 result per column, de-skewed to a single valid cycle

**Input skewing:** Row `k` receives its activation `k` cycles later than row 0. For M output rows, total streaming ticks = `M + ROWS + COLS − 1` instead of `M × (ROWS + COLS)`, giving near-100% MAC utilization as M grows.

### PE (`pe.sv`)

```
         weight_ld
             │  en
             ▼  ▼
          ┌──────────┐
          │    PE    ├──► data_out
          │  weight  │
data_in──►│  (reg)   │
          │  × + acc │
acc_in ──►│          ├──► acc_out
          └──────────┘
```

- Phase 1 (`weight_ld=1, en=0`): latch weight into PE register
- Phase 2 (`weight_ld=0, en=1`): stream activations, accumulate partial sums
- `int8 × int8 → int32`, then `int32 + int32 → int32`

## Running a Real Model: QuickDraw Sketch Classifier

`scripts/train_quickdraw.py` downloads a subset of the [Google QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) dataset (28×28 hand-drawn sketches — same bitmap format as MNIST), trains a three-layer MLP, and exports a dynamically-quantized ONNX model. `scripts/run_quickdraw.py` then runs it end-to-end through the TinyXPU EP.

A pure MLP is the ideal target for this hardware: every compute-heavy operation is `MatMulInteger` on the systolic array, and `Relu` is now also claimed by the EP — **the CPU EP handles nothing from the model's forward pass.**

### Hardware ReLU

The systolic array has a `relu_en` input. When asserted, the output stage clamps any negative `acc_out` value to zero in the same cycle (purely combinational, zero latency). The EP drives `relu_en=1` when a fused `MatMulInteger+Relu` kernel is compiled.

### Training and export

```sh
source .venv/bin/activate
python scripts/train_quickdraw.py          # downloads data, trains, exports
```

This produces:
- `quickdraw.onnx` — float32 model
- `quickdraw-int8.onnx` — dynamically quantized (QInt8 weights)

**Dynamic int8 quantization** replaces each `MatMul` with an integer kernel:

```
DynamicQuantizeLinear(activation) ──► (a_int8, a_scale, a_zero_point)
                                              │
MatMulInteger ──────────────────────────────►│◄── w_int8  (pre-quantized weight)
        │
        ▼  int32
Cast → float32
        │
Mul(combined_scale) → Add(bias) → Relu → …
```

### What the hardware sees vs what the CPU handles

| Layer | Operator | Handled by |
|-------|----------|------------|
| FC1 (784→512) | `MatMulInteger` | **TinyXPU** (systolic array) |
| ReLU | `Relu` | **TinyXPU** (EP C++ kernel) |
| FC2 (512→256) | `MatMulInteger` | **TinyXPU** (systolic array) |
| ReLU | `Relu` | **TinyXPU** (EP C++ kernel) |
| FC3 (256→10) | `MatMulInteger` | **TinyXPU** (systolic array) |
| Dequantize | `DynamicQuantizeLinear`, `Cast`, `Mul`, `Add` | CPU EP |
| Output | `Softmax` | CPU EP |

### Running the classifier

```sh
source .venv/bin/activate
python scripts/run_quickdraw.py
```

### Tiling

The default 16×16 array tiles each matrix multiply into `⌈K/16⌉ × ⌈N/16⌉` blocks. For FC1 (784→512) this is `49 × 32 = 1568` hardware passes. Rebuilding with larger `-DSIM_ROWS` / `-DSIM_COLS` reduces pass count proportionally.

## Related Projects

- [tiny-tpu-v2/tiny-tpu](https://github.com/tiny-tpu-v2/tiny-tpu/tree/main) — 2×2 matmul + ReLU
- [Alanma23/tinytinyTPU](https://github.com/Alanma23/tinytinyTPU) — 2×2 matmul + ReLU/ReLU6
