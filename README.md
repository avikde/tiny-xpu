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
pip install matplotlib cocotb onnxruntime==1.24.2 onnx "optimum[onnxruntime]"
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

## Running a Real Model: BERT-tiny

`scripts/run_bert_tiny.py` downloads [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny) from HuggingFace, quantizes it, and runs it end-to-end through the TinyXPU EP.

### Model generation

The script follows the standard industry workflow for deploying a transformer on an int8 accelerator:

```
prajjwal1/bert-tiny (PyTorch, float32)
        │
        │  optimum-cli export onnx          # torch.export → ONNX opset 17
        ▼
bert-tiny-float/model.onnx
        │
        │  optimum-cli onnxruntime quantize  # dynamic int8 quantization
        ▼
bert-tiny-int8/model_quantized.onnx
```

**Dynamic int8 quantization** leaves the graph structure intact but replaces each `Gemm`/`MatMul` with:

```
DynamicQuantizeLinear(activation) ──► (a_int8, a_scale, a_zero_point)
                                              │
MatMulInteger(a_int8, w_int8) ──────────────►│◄── w_int8  (pre-quantized weight)
        │
        ▼  int32
Cast → float32
        │
Mul(combined_scale) → DequantizeLinear → Add(bias) → …
```

BERT-tiny has 2 encoder layers, each with 6 linear projections (Q, K, V, attention output, and 2 FFN layers), plus a pooler — **16 `MatMulInteger` nodes** in total. Everything else (LayerNorm, Softmax, GELU, Gather for embeddings, residual Add) runs on the ONNX Runtime CPU EP unmodified.

### Tiling: fitting large matrices onto a small array

BERT-tiny's hidden size is 128 and intermediate size is 512, so the linear layers produce matrices like [seq, 128] × [128, 128] and [seq, 128] × [128, 512] — far larger than the default 16×16 systolic array.

The EP tiles each `MatMulInteger` into `⌈K/ROWS⌉ × ⌈N/COLS⌉` blocks and accumulates partial int32 sums:

```
C[M, N] = 0
for k₀ in 0, ROWS, 2·ROWS, …, K:
    for n₀ in 0, COLS, 2·COLS, …, N:
        load  B[k₀:k₀+ROWS, n₀:n₀+COLS] as weights
        stream A[:, k₀:k₀+ROWS]
        C[:, n₀:n₀+COLS] += systolic_output   # int32 accumulate
```

For the [16 × 128] × [128 × 128] case with a 16×16 array this is 64 hardware passes (8 K-tiles × 8 N-tiles). Rebuilding with `-DSIM_ROWS=128 -DSIM_COLS=128` reduces it to a single pass.

### What the hardware sees vs what the CPU handles

| Layer type | Operator | Handled by |
|------------|----------|------------|
| Q / K / V projections | `MatMulInteger` | **TinyXPU (tiled)** |
| Attention output / FFN | `MatMulInteger` | **TinyXPU (tiled)** |
| Activation quantization | `DynamicQuantizeLinear` | CPU EP |
| Attention scores | `Softmax`, `Div`, `Mul` | CPU EP |
| Layer norm | `LayerNormalization` | CPU EP |
| Residual add | `Add` | CPU EP |
| Token / position embeddings | `Gather` | CPU EP |

## Related Projects

- [tiny-tpu-v2/tiny-tpu](https://github.com/tiny-tpu-v2/tiny-tpu/tree/main) — 2×2 matmul + ReLU
- [Alanma23/tinytinyTPU](https://github.com/Alanma23/tinytinyTPU) — 2×2 matmul + ReLU/ReLU6
