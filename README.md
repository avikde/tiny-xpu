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
- `weight_in_top[COLS]`, `weight_in` — load weights via systolic cascade (weight_in=1 for ROWS cycles)
- `acc_out[COLS]` — raw int32 result per column (no de-skew)
- `q_out[COLS]` — int8 requantized output (valid when `requant_en=1`)

**Input skewing:** Row `k` receives its activation `k` cycles later than row 0. For M output rows, total streaming ticks = `M + ROWS + COLS − 1` instead of `M × (ROWS + COLS)`, giving near-100% MAC utilization as M grows.

### PE (`pe.sv`)

```
         weight_in
             │  en
             ▼  ▼
          ┌──────────┐
          │    PE    ├──► data_out
          │  weight  │
data_in──►│  (reg)   │
          │  × + acc │
acc_in ──►│          ├──► acc_out
          └──────────┘
             │
             ▼
         weight_out
```

TO FIX:

- `relu_en` — clamp negative `acc_out` values to zero (combinational)
- `requant_en`, `bias_in[COLS]`, `M0`, `rshift`, `zero_pt` — drive the requantization stage

### Usage for matrix multiplication

([Background on systolic matrix multiplication](https://www.avikde.me/p/systolic-arrays-for-general-robotics))

**Input staging (M rows, R = HW_ROWS):**
- **Bias**: Enters from top, staggered by row: `b(1,1)` at t=1, `b(2,1)` at t=2, ..., `b(m,1)` at t=M
- **Activations**: Enters from left, same stagger: `x(1,1)` at t=1 (top PE), `x(2,1)` at t=2, ..., `x(m,1)` at t=M

**Pipeline timing:**
- At `t > M+K`, the top-left PE finishes the first matrix product
- At `t = M+K+1`, that PE is idle and *could* accept a new weight

### Design enhancements for weight loading

In [TPU-like](https://arxiv.org/pdf/1704.04760) architectures, weight loading is done in a separate phase and needs the pipeline to fully drain. This adds latency. Quoting the TPU v1 paper,

> The weights are preloaded, and take effect with the advancing wave alongside the first data of a new block

> About 35% of cycles are spent waiting for weights to load from memory into the matrix unit, which occurs during the 4 fully connected layers that run at an operational intensity of just 32

The time for a `(M,K) × (K,N)` product is `M+R+N` cycles (`R` cycles to fill the pipeline, `M` cycles of compute, `N` cycles to drain). With separate-phase weight loading, you must drain the pipeline and reload: tile-to-tile **latency is `M+K+R+N`** cycles.

#### 1. Pipelined sequential weight loading

**Idea:** Add a `weight_in` signal, so that PEs distinguish the weight from data (bias/partial sums). When `weight_in=1`, a weight entering from the top on acc_in triggers a load-and-forward chain that fills the column while computation tails out.

**PE behavior:**
- `weight_in=1`: Latch as new weight, reset accumulator to 0, pass tagged weight down immediately
- `weight_in=0`: Add to accumulator (first untagged is bias, subsequent are partial sums)
- `weight_out` is set to `weight_in` so that it is received by the south PE in the next cycle.

For the matrix product, it takes `M+K` cycles from the first input entry to the start of weight loading for the next product. The next product can start immediately after the first new weight column is loaded over `K` cycles. Therefore, the tile-to-tile **latency is `M+2K` cycles**.

**Hardware tradeoff:** Extra bit on each north-south connection for the tag.

#### 2. Double-buffered weights

**Idea:** Keep two weight registers per PE (active and shadow). Load the next tile's weights into the shadow buffer during computation, then swap at the tile boundary.

The switch propagates diagonally, catching each PE just as it becomes idle. PE(1,1) starts the new tile immediately after finishing its previous row, while bottom-right PEs finish the old tile using their (still-active) old weights.

The MACs should be able to start right after each other, leading to a **latency of `M+K`** cycles.

Other systems:
- [Tiny-TPU](https://www.tinytpu.com/) uses the same propagating control pattern (switch + accept) rather than data tagging, achieving continuous inference without the ~35% idle time from separate load phases.
- Apple Neural Engine is effectively double-buffered via SRAM banks.

**Hardware tradeoff:**
- **2× weight registers per PE** (~16 bits vs 8 bits)
- **Separate weight cascade** (cannot reuse `acc_in`—it's busy with partial sums)
- **2 control bits** (switch + accept flags)
- More control complexity, but no tag bit on data paths

### Design enhancement: output taps

E.g. with a weight-stationary array with 16 rows, let's say the weight matrix W has 8 rows.
```
W
---
0
---> normal output after 16 rows
```
suffers the latency all 16 rows. But if we had a few extra output taps (let’s say after 8 rows in this example)
```
W
---> new output tap after 8 rows
0
---> normal output after 16 rows
```
then we could be done after 8 cycles. With a bit more work we could even run two products in parallel (relevant for multi-headed attention) to get back full utilization.
```
W1
---> new output tap after 8 rows
W2
---> normal output after 16 rows
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
