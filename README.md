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
- `pip install cocotb` -- Python tool for more powerful testing capabilities

Build:

```shell
mkdir -p build && cd build
cmake ..
make -j
```

Test:

```shell
cd build && ctest --verbose
```

Tests produce waveform files (`*.fst`) in `test/sim_build/`. Open them in VSCode with the Surfer extension to inspect signals.

## Supported operations

For this project, our goal was to develop something broadly applicable. 

We initially focused on General Matrix Multiply (GEMM); because it parameterizes the scaling constants `α` and `β` (`C = αAB + βC`), it subsumes pure matrix multiplication as a special case and enables fused multiply-accumulate patterns that avoid redundant memory writes. Many types of computations common in deep learning, including matrix multiplication, are [specializations of a GEMM operation](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html). GEMMs show up in [dense fully connected networks](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html) that are a core component of transformers and RNNs. Lastly, it is one of the [Basic Linear Algebra Subproblems (BLAS)](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) that have been the bedrock of scientific computing.

As a step toward GEMM, we add a MatMul.

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

This is why it's called "weight-stationary" — weights move once, data flows repeatedly

## Related projects

There are a number of "tiny TPU"-type projects, due to the current popularity of TPUs and LLMs.

- [tiny-tpu-v2/tiny-tpu](https://github.com/tiny-tpu-v2/tiny-tpu/tree/main) - 2x2 matmul + ReLU to solve XOR problem
- [Alanma23/tinytinyTPU](https://github.com/Alanma23/tinytinyTPU) - 2x2 matmul + ReLU / ReLU6

