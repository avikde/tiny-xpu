# tiny-xpu

## Project goal

While there are other projects building up small (~2x2) TPU-inspired designs (see related projects below), this project has a salient combination of goals:

- Modular SystemVerilog setup to support non-rectangular systolic architectures
- Easy software interface via ONNX EP and maybe others
- Support for FPGA deployment

## Setup

Set up in WSL or other Linux: 

- `sudo apt install iverilog` -- Icarus Verilog for simulation
- Install the [Surfer waveform viewer](https://marketplace.visualstudio.com/items?itemName=surfer-project.surfer) VSCode extension for viewing `.vcd` waveform files
- `sudo apt install yosys` -- Yosys for synthesis (or [build from source](https://github.com/YosysHQ/yosys) for the latest version)
- `pip install cocotb` -- Python tool for more powerful testing capabilities

## Building

```shell
mkdir -p build && cd build
cmake ..
make -j
```

## Testing

```shell
cd build && ctest --verbose
```

Tests produce waveform files (`*.fst`) in `test/sim_build/`. Open them in VSCode with the Surfer extension to inspect signals.

## Related projects

There are a number of "tiny TPU"-type projects, due to the current popularity of TPUs and LLMs.

- [tiny-tpu-v2/tiny-tpu](https://github.com/tiny-tpu-v2/tiny-tpu/tree/main)
- [Alanma23/tinytinyTPU](https://github.com/Alanma23/tinytinyTPU)

