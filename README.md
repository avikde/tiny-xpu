# tiny-xpu

## Project goal

While there are other projects building up small (~2x2) TPU-inspired designs (see related projects below), this project has a salient combination of goals:

- Modular SystemVerilog setup to support non-rectangular systolic architectures
- Easy software interface via ONNX EP and maybe others
- Support for FPGA deployment

## Building

```shell
mkdir -p build && cd build
cmake ..
make -j
```

## Related projects

There are a number of "tiny TPU"-type projects, due to the current popularity of TPUs and LLMs.

- [tiny-tpu-v2/tiny-tpu](https://github.com/tiny-tpu-v2/tiny-tpu/tree/main)
- [Alanma23/tinytinyTPU](https://github.com/Alanma23/tinytinyTPU)

