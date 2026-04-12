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
pip install matplotlib cocotb onnxruntime==1.24.2 onnx torch torchvision
```

## Build and Test

```sh
mkdir -p build && cd build
cmake .. -DSIM=ON   # SIM=ON links Verilator into the ONNX EP
make -j
ctest --verbose     # waveforms written to test/sim_build/*.fst
```

Key CMake flags:
- `-DSIM=ON` — use Verilator simulation backend (required for software runs)
- `-DSIM_ROWS=N -DSIM_COLS=N` — override array size (default **64×64**)

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
- `acc_out[COLS]` — raw int32 result per column (no de-skew)
- `relu_en` — clamp negative `acc_out` values to zero (combinational)
- `requant_en`, `bias_in[COLS]`, `M0`, `rshift`, `zero_pt` — drive the requantization stage
- `q_out[COLS]` — int8 requantized output (valid when `requant_en=1`)

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

### Requantization stage (`requant.sv`)

A COLS-wide combinational output stage that converts the int32 accumulator directly to int8, so the next layer can consume it without any float dequantization.

```
acc_int32 ──► + bias ──► × M0 (63-bit) ──► >>> rshift ──► + zero_pt ──► sat8 ──► [ReLU] ──► q_int8
```

The combined scale `S = a_scale × w_scale / y_scale` is pre-computed by the EP and encoded as a fixed-point multiplier `M0 = round(S × 2³¹)` with `rshift = 31`. This is the standard fixed-point requantization used by integer-only NPUs (TFLite, ONNX Runtime, Apple Neural Engine, Qualcomm Hexagon).

## Running a Real Model: QuickDraw Sketch Classifier

`scripts/train_quickdraw.py` downloads a subset of the [Google QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) dataset (10 sketch categories, 28×28 bitmaps), applies static post-training quantization, and exports a fully-integer ONNX model. `scripts/run_quickdraw.py` runs it end-to-end through the TinyXPU EP.

### Network and array co-design

The network is sized so that **no layer requires tiling**:

| Dimension | Value |
|-----------|-------|
| Input | 28×28 → 8×8 area-average → **64 features** |
| FC1 | 64 → 64 + ReLU |
| FC2 | 64 → 32 + ReLU |
| FC3 | 32 → 10 (logits) |
| Array size | **64×64** |

All inner dimensions (K ≤ 64, N ≤ 64) fit in one hardware pass — no tiling, no K-accumulation across passes.

### Fully-integer pipeline

Static PTQ (static post-training quantization) produces a `QLinearMatMul` graph where every scale and zero-point is a baked-in constant. The hardware executes each layer as a single fused operation:

```
int8 activations
      │
      ▼
 Systolic array (int8×int8 → int32 accumulation)
      │
      ▼
 Requant stage: int32 → int8
   biased  = acc + bias
   product = biased × M0   (63-bit fixed-point)
   shifted = product >>> 31
   q_out   = sat8(shifted + zero_pt)  [+ ReLU if fused]
      │
      ▼
 int8 activations  ──► next layer
```

There is **no float32 between layers**. `acc_out` (int32) and `q_out` (int8) are both driven by the array in the same clock cycle; the EP reads `q_out` directly.

### What the hardware sees vs what the CPU handles

| Layer | Operator | Handled by |
|-------|----------|------------|
| FC1 (64→64) + ReLU | `QLinearMatMul` | **TinyXPU** (systolic array + requant stage) |
| FC2 (64→32) + ReLU | `QLinearMatMul` | **TinyXPU** (systolic array + requant stage) |
| FC3 (32→10) | `QLinearMatMul` | **TinyXPU** (systolic array + requant stage) |
| Output | `ArgMax` / host readout | CPU EP |

### Training and export

```sh
source .venv/bin/activate
python scripts/train_quickdraw.py          # downloads data, trains, exports
```

This produces:
- `quickdraw.onnx` — float32 model
- `quickdraw-int8.onnx` — statically quantized (`QLinearMatMul` nodes with embedded scales)

**Static PTQ workflow:**
1. Train float32 model (15 epochs, Adam, CrossEntropy)
2. Fuse `Linear + ReLU` pairs
3. Insert `QuantStub` / `DeQuantStub`, configure per-tensor int8 qconfig
4. Calibration pass on training set → collect activation statistics
5. `convert()` → `QLinearMatMul` nodes with constant `a_scale`, `b_scale`, `y_scale`, `y_zero_point`

### Running the classifier

```sh
source .venv/bin/activate
python scripts/run_quickdraw.py
```

## Interactive MLP Explorer

A browser-based tool for exploring the depth/width tradeoff from two perspectives simultaneously: ML expressivity and systolic array efficiency. Hosted at [avikde.github.io/tiny-xpu](https://avikde.github.io/tiny-xpu/).

### How to use

Pick a **task** (spiral classification or sum-of-sines regression), then adjust **depth** (1–8 hidden layers) and **width** (4–128 neurons). The left panel trains a small network in-browser and shows the decision boundary or fitted curve. The right panel shows the hardware profile on a configurable systolic array.

The key experiment: find two configurations with the same parameter count but different depth/width. The ML accuracy will be similar — the hardware profile will not.

### Hardware metrics

| Metric | Formula |
|--------|---------|
| Spatial utilization | `min(width, array_cols) / array_cols` |
| Temporal utilization | `M / (M + ROWS + N − 2)` — pipeline fill, M=64 batch |
| Throughput | `spatial × temporal × ROWS × COLS` MACs/cycle |
| Inference latency | `depth × (M + ROWS + N − 2)` cycles |
| Arithmetic intensity | `M·N / (N + M)` MACs/byte (weight-stationary) |

The roofline diagram shows whether the current configuration is memory-bound or compute-bound given a 16 B/cycle SRAM bandwidth assumption.

### Running locally

```sh
node web/serve.js   # serves web/ at http://localhost:3000
```

## Related Projects

- [tiny-tpu-v2/tiny-tpu](https://github.com/tiny-tpu-v2/tiny-tpu/tree/main) — 2×2 matmul + ReLU
- [Alanma23/tinytinyTPU](https://github.com/Alanma23/tinytinyTPU) — 2×2 matmul + ReLU/ReLU6
