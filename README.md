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
python scripts/run_bert_tiny.py   # download prajjwal1/bert-tiny, quantize, run on EP
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

## Running a Real Model: MNIST LeNet

`scripts/run_mnist.py` downloads the [ONNX Model Zoo MNIST](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist) LeNet model, applies dynamic int8 quantization, and runs it end-to-end through the TinyXPU EP.

This model is an ideal pedagogical target: it is a well-known, pretrained classifier whose compute is dominated by operations the systolic array handles natively, with no changes required to the SystemVerilog.

### Model preparation

```
mnist-12.onnx  (ONNX Model Zoo, float32 LeNet)
        │
        │  onnxruntime.quantization.quantize_dynamic   # dynamic int8 quantization
        ▼
mnist-int8.onnx
```

**Dynamic int8 quantization** replaces each `Conv`/`MatMul` with an integer kernel plus dequantization:

```
DynamicQuantizeLinear(activation) ──► (a_int8, a_scale, a_zero_point)
                                              │
ConvInteger / MatMulInteger ────────────────►│◄── w_int8  (pre-quantized weight)
        │
        ▼  int32
Cast → float32
        │
Mul(combined_scale) → Add(bias) → …
```

### Mapping convolutions to the systolic array

Systolic arrays are the standard hardware primitive for convolution because convolution *is* matrix multiplication in disguise. The EP performs an **im2col** transform in C++ to convert each convolution into a `MatMulInteger` call, requiring no changes to the SystemVerilog:

```
Input patch (H_out × W_out, C_in × kH × kW)  ←  im2col
Weight matrix (C_out, C_in × kH × kW)ᵀ
        │
        ▼
MatMulInteger on systolic array → output feature map (H_out × W_out, C_out)
```

### What the hardware sees vs what the CPU handles

| Layer | Operator | Handled by |
|-------|----------|------------|
| Conv1 (5×5, 1→20) | `ConvInteger` → im2col → `MatMulInteger` | **TinyXPU** |
| Conv2 (5×5, 20→50) | `ConvInteger` → im2col → `MatMulInteger` | **TinyXPU** |
| FC1 (800→500) | `MatMulInteger` | **TinyXPU** |
| FC2 (500→10) | `MatMulInteger` | **TinyXPU** |
| ReLU | `Relu` | EP (C++ post-processing) |
| Pooling | `MaxPool` | CPU EP |
| Output | `Softmax` | CPU EP |

### Tiling

The default 16×16 array tiles each matrix multiply into `⌈K/16⌉ × ⌈N/16⌉` blocks. For FC1 (800→500) this is `50 × 32 = 1600` hardware passes. Rebuilding with larger `-DSIM_ROWS` / `-DSIM_COLS` reduces pass count proportionally.

## Related Projects

- [tiny-tpu-v2/tiny-tpu](https://github.com/tiny-tpu-v2/tiny-tpu/tree/main) — 2×2 matmul + ReLU
- [Alanma23/tinytinyTPU](https://github.com/Alanma23/tinytinyTPU) — 2×2 matmul + ReLU/ReLU6
