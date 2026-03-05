"""
Generate a MatMulInteger ONNX model for TinyXPU (4x4 int8 systolic array).

The model computes Y = X @ W where:
  X: runtime input,  int8, shape [M, 4]  (M rows, K=4 columns)
  W: baked-in weight, int8, shape [4, 4]  (K=4 rows, N=4 columns)
  Y: output,          int32, shape [M, 4]

Constraints match the hardware: K == ROWS == 4, N == COLS == 4.
M is unrestricted — the systolic array streams one row at a time.

Usage:
    python scripts/matmul.py
Output:
    scripts/matmul_integer_4x4.onnx
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx import numpy_helper

def make_matmul_graph(W_data):
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [None, W_data.shape[0]])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [None, W_data.shape[1]])
    W_init = numpy_helper.from_array(W_data, name="W")
    node = helper.make_node("MatMulInteger", inputs=["X", "W"], outputs=["Y"])
    graph = helper.make_graph(
        [node],
        "MatMulInteger_4x4",
        [X],
        [Y],
        initializer=[W_init],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21)],
        producer_name="tiny-xpu",
    )
    model.ir_version = 10

    onnx.checker.check_model(model)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"matmul_integer_{W_data.shape[0]}x{W_data.shape[1]}.onnx")
    onnx.save(model, out_path)
    print(f"Model saved to {out_path}")
    print(f"  Input X : int8  [M, {W_data.shape[0]}]  (M is runtime batch size)")
    print(f"  Weight W: int8  [{W_data.shape[0]}, {W_data.shape[1]}]  (baked in)")
    print(f"  Output Y: int32 [M, {W_data.shape[1]}]")

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    W1_data = rng.integers(1, 6, size=(16, 4), dtype=np.int8)
    W2_data = rng.integers(1, 6, size=(8, 8), dtype=np.int8)
    W3_data = rng.integers(1, 6, size=(4, 16), dtype=np.int8)
    make_matmul_graph(W1_data)
    make_matmul_graph(W2_data)
    make_matmul_graph(W3_data)
