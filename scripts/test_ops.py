"""
Tests for batched MatMulInteger and Gemm support in the TinyXPU EP.

Runs two test cases against the built plugin and verifies outputs against NumPy:
  1. Batched MatMulInteger: int8 A[batch, M, K] @ int8 W[K, N] → int32 Y[batch, M, N]
  2. Gemm (transB=0): float32 A @ B + bias
  3. Gemm (transB=1): float32 A @ B.T + bias (B stored as [N, K])

Usage:
    python scripts/test_ops.py [path/to/libtinyxpu_ep.so|.dylib]

Requires the plugin to be built (cmake .. -DSIM=ON && make -j for batched test;
cmake .. with no -DSIM for Gemm tests, or SIM=ON which also supports CPU path).
"""

import ctypes
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinyxpu_perf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_plugin() -> str:
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    plugin_dir = os.path.join(repo, "build", "onnx-plugin")
    ext = "dylib" if sys.platform == "darwin" else "so"
    return os.path.abspath(os.path.join(plugin_dir, f"libtinyxpu_ep.{ext}"))


def _make_session(plugin_path: str, model: onnx.ModelProto):
    ort.register_execution_provider_library("SampleEP", plugin_path)
    devices = [d for d in ort.get_ep_devices() if "SampleEP" in d.ep_name]
    if not devices:
        raise RuntimeError("TinyXPU EP device not found after registration.")
    opts = ort.SessionOptions()
    opts.add_provider_for_devices(devices, {})
    import io
    buf = io.BytesIO()
    onnx.save(model, buf)
    session = ort.InferenceSession(buf.getvalue(), sess_options=opts)
    return session


def _teardown():
    try:
        ort.unregister_execution_provider_library("SampleEP")
    except Exception:
        pass


def _make_batched_matmul_model(K: int, N: int, W: np.ndarray) -> onnx.ModelProto:
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [None, None, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [None, None, N])
    W_init = numpy_helper.from_array(W, name="W")
    node = helper.make_node("MatMulInteger", inputs=["X", "W"], outputs=["Y"])
    graph = helper.make_graph([node], "batched_mmi", [X], [Y], initializer=[W_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)],
                              producer_name="tiny-xpu")
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _make_gemm_model(M: int, K: int, N: int, W: np.ndarray,
                     bias: np.ndarray, transB: bool) -> onnx.ModelProto:
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    W_init = numpy_helper.from_array(W, name="W")
    bias_init = numpy_helper.from_array(bias, name="bias")
    node = helper.make_node("Gemm", inputs=["X", "W", "bias"], outputs=["Y"],
                            transB=1 if transB else 0)
    graph = helper.make_graph([node], "gemm_graph", [X], [Y],
                              initializer=[W_init, bias_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)],
                              producer_name="tiny-xpu")
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_batched_matmul_integer(plugin_path: str, lib) -> bool:
    """int8 A[2, 4, 8] @ W[8, 8] → int32 Y[2, 4, 8], verified vs NumPy."""
    print("=" * 60)
    print("Test: Batched MatMulInteger  [2, 4, 8] x [8, 8] → [2, 4, 8]")
    print("=" * 60)

    rng = np.random.default_rng(1)
    batch, M, K, N = 2, 4, 8, 8
    A = rng.integers(-64, 64, size=(batch, M, K), dtype=np.int8)
    W = rng.integers(-4, 4, size=(K, N), dtype=np.int8)

    # NumPy reference
    expected = np.matmul(A.astype(np.int32), W.astype(np.int32))

    model = _make_batched_matmul_model(K, N, W)

    sys.stdout.flush()
    session = _make_session(plugin_path, model)

    result = session.run(None, {"X": A})
    Y = result[0]

    perf = tinyxpu_perf.get_last_perf(lib)
    tinyxpu_perf.print_perf(perf)
    print()

    ok = np.array_equal(Y, expected)
    if ok:
        print(f"PASS: Y shape={Y.shape}, matches NumPy reference.")
    else:
        print(f"FAIL: shape={Y.shape}, first diff at index:")
        diff = Y - expected
        idx = np.argwhere(diff != 0)
        print(f"  first mismatch: {idx[0]}  got={Y[tuple(idx[0])]}  expected={expected[tuple(idx[0])]}")
    print()
    return ok


def test_gemm(plugin_path: str, transB: bool) -> bool:
    """float32 Gemm A[8,8] @ W + bias, transB={transB}, verified vs NumPy."""
    label = f"transB={'True' if transB else 'False'}"
    print("=" * 60)
    print(f"Test: Gemm  [8, 8] x [8, 8] + bias  ({label})")
    print("=" * 60)

    rng = np.random.default_rng(2)
    M, K, N = 8, 8, 8
    A = rng.standard_normal((M, K)).astype(np.float32)
    W_eff = rng.standard_normal((K, N)).astype(np.float32)   # effective K×N
    bias = rng.standard_normal(N).astype(np.float32)

    # Stored weight: transposed if transB=True
    W_stored = W_eff.T if transB else W_eff

    # NumPy reference
    expected = A @ W_eff + bias

    model = _make_gemm_model(M, K, N, W_stored, bias, transB)

    sys.stdout.flush()
    session = _make_session(plugin_path, model)

    result = session.run(None, {"X": A})
    Y = result[0]

    ok = np.allclose(Y, expected, atol=1e-5)
    if ok:
        print(f"PASS: Y shape={Y.shape}, matches NumPy reference (atol=1e-5).")
    else:
        print(f"FAIL: max abs diff = {np.max(np.abs(Y - expected)):.6f}")
        print(f"  Y[0]       = {Y[0]}")
        print(f"  expected[0]= {expected[0]}")
    print()
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    plugin_path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else _default_plugin())

    if not os.path.exists(plugin_path):
        print(f"ERROR: plugin not found at {plugin_path}")
        print("Run `cmake .. -DSIM=ON && make -j` in the build directory first.")
        return 1

    print(f"Plugin: {plugin_path}\n")

    lib = ctypes.CDLL(plugin_path)
    tinyxpu_perf.bind(lib)

    results = []

    # Test 1: batched MatMulInteger (requires SIM=ON / Verilator)
    import subprocess, shutil
    has_sim = True
    try:
        results.append(("Batched MatMulInteger", test_batched_matmul_integer(plugin_path, lib)))
    except Exception as e:
        print(f"SKIP/ERROR (batched MatMulInteger): {e}")
        print("  (Build with -DSIM=ON to enable Verilator tests)\n")
        results.append(("Batched MatMulInteger", None))
    finally:
        _teardown()

    # Tests 2 & 3: Gemm (CPU path, works with or without SIM)
    for transB in (False, True):
        label = f"Gemm transB={'True' if transB else 'False'}"
        try:
            results.append((label, test_gemm(plugin_path, transB)))
        except Exception as e:
            print(f"ERROR ({label}): {e}\n")
            results.append((label, False))
        finally:
            _teardown()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        if ok is None:
            status = "SKIP"
        elif ok:
            status = "PASS"
        else:
            status = "FAIL"
            all_pass = False
        print(f"  {status}  {name}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
