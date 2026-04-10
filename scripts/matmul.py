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

def make_batched_matmul_integer_graph(W_data):
    """MatMulInteger with 3-D input X=[batch, M, K] and 2-D weight W=[K, N]."""
    K, N = W_data.shape
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [None, None, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [None, None, N])
    W_init = numpy_helper.from_array(W_data, name="W")
    node = helper.make_node("MatMulInteger", inputs=["X", "W"], outputs=["Y"])
    graph = helper.make_graph(
        [node],
        "BatchedMatMulInteger",
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
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"matmul_integer_batched_{K}x{N}.onnx",
    )
    onnx.save(model, out_path)
    print(f"Model saved to {out_path}")
    print(f"  Input X : int8  [batch, M, {K}]")
    print(f"  Weight W: int8  [{K}, {N}]  (baked in)")
    print(f"  Output Y: int32 [batch, M, {N}]")


def make_gemm_graph(M, K, N, transB=False):
    """Gemm: Y = X @ W + bias  (float32). transB=True stores W as [N, K]."""
    rng = np.random.default_rng(0)
    W_shape = (N, K) if transB else (K, N)
    W_data = rng.standard_normal(W_shape).astype(np.float32)
    bias_data = rng.standard_normal(N).astype(np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    W_init = numpy_helper.from_array(W_data, name="W")
    bias_init = numpy_helper.from_array(bias_data, name="bias")
    node = helper.make_node(
        "Gemm",
        inputs=["X", "W", "bias"],
        outputs=["Y"],
        transB=1 if transB else 0,
    )
    graph = helper.make_graph(
        [node],
        "Gemm_graph",
        [X],
        [Y],
        initializer=[W_init, bias_init],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21)],
        producer_name="tiny-xpu",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    tb = "_transB" if transB else ""
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"gemm_{M}x{K}x{N}{tb}.onnx",
    )
    onnx.save(model, out_path)
    print(f"Model saved to {out_path}")
    print(f"  Input X : float32  [{M}, {K}]")
    print(f"  Weight W: float32  {list(W_shape)}  (baked in, transB={transB})")
    print(f"  Bias    : float32  [{N}]  (baked in)")
    print(f"  Output Y: float32  [{M}, {N}]")


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    make_matmul_graph(rng.integers(1, 6, size=(16, 4), dtype=np.int8))
    make_matmul_graph(rng.integers(1, 6, size=(8, 8), dtype=np.int8))
    make_matmul_graph(rng.integers(1, 6, size=(4, 16), dtype=np.int8))
    make_matmul_graph(rng.integers(1, 6, size=(16, 16), dtype=np.int8))
    make_batched_matmul_integer_graph(rng.integers(1, 6, size=(8, 8), dtype=np.int8))
    make_gemm_graph(M=8, K=8, N=8, transB=False)
    make_gemm_graph(M=8, K=8, N=8, transB=True)
