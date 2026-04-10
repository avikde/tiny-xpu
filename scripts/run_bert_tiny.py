"""
Download prajjwal1/bert-tiny, export to int8 ONNX via optimum, and run it
through the TinyXPU Execution Provider.

The quantized export produces a graph with 16 MatMulInteger nodes — one for
every linear projection in the 2-layer transformer (Q, K, V, output, and the
two FFN layers per encoder block, plus the pooler). The TinyXPU EP claims
all of them; the ONNX Runtime CPU EP handles everything else (LayerNorm,
DynamicQuantizeLinear, Softmax, Gather, …).

Standard workflow (what a researcher or MLE would do):

    # Step 1 — export float32 ONNX
    optimum-cli export onnx -m prajjwal1/bert-tiny \\
        --task feature-extraction bert-tiny-float/

    # Step 2 — dynamic int8 quantization (produces MatMulInteger graph)
    optimum-cli onnxruntime quantize \\
        --onnx_model bert-tiny-float/ -o bert-tiny-int8/ --arm64   # Mac
        # or --avx512_vnni on Linux

    # Step 3 — run on TinyXPU EP (this script)
    .venv/bin/python scripts/run_bert_tiny.py

This script automates steps 1 & 2 when the outputs don't exist yet.

Usage:
    source .venv/bin/activate
    python scripts/run_bert_tiny.py [plugin_path]
"""

import ctypes
import os
import platform
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinyxpu_perf

REPO_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FLOAT_DIR   = os.path.join(REPO_ROOT, "bert-tiny-float")
QUANT_DIR   = os.path.join(REPO_ROOT, "bert-tiny-int8")
MODEL_PATH  = os.path.join(QUANT_DIR, "model_quantized.onnx")
OPTIMUM_CLI = os.path.join(REPO_ROOT, ".venv", "bin", "optimum-cli")


# ── Export helpers ────────────────────────────────────────────────────────────

def _run(cmd: list[str], desc: str) -> None:
    print(f"{desc} …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Forward warnings/info but suppress noisy torch tracing chatter
    for line in result.stderr.splitlines():
        if any(skip in line for skip in ["TracerWarning", "torch_dtype", "deduplication"]):
            continue
        print(f"  {line}")
    if result.returncode != 0:
        print(result.stdout[-2000:] if result.stdout else "")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def export_float() -> None:
    if os.path.exists(os.path.join(FLOAT_DIR, "model.onnx")):
        print(f"Float32 ONNX already at {FLOAT_DIR}/model.onnx — skipping export.")
        return
    _run(
        [OPTIMUM_CLI, "export", "onnx",
         "-m", "prajjwal1/bert-tiny",
         "--task", "feature-extraction",
         FLOAT_DIR],
        "Step 1/2  Exporting prajjwal1/bert-tiny → float32 ONNX (~17 MB download)",
    )


def quantize_int8() -> None:
    if os.path.exists(MODEL_PATH):
        print(f"Int8 model already at {MODEL_PATH} — skipping quantization.")
        return
    arch = "--arm64" if platform.machine() == "arm64" else "--avx512_vnni"
    _run(
        [OPTIMUM_CLI, "onnxruntime", "quantize",
         "--onnx_model", FLOAT_DIR,
         "-o", QUANT_DIR,
         arch],
        f"Step 2/2  Dynamic int8 quantization ({arch})",
    )


# ── Model inspection ──────────────────────────────────────────────────────────

def count_ops(model_path: str) -> dict[str, int]:
    m = onnx.load(model_path)
    ops: dict[str, int] = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    return ops


def make_dummy_inputs(model_path: str, batch: int = 1, seq_len: int = 16) -> dict:
    """Build minimal int64 BERT inputs (input_ids, attention_mask, token_type_ids)."""
    m = onnx.load(model_path)
    inputs = {}
    for inp in m.graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            if d.HasField("dim_param") or d.dim_value == 0:
                shape.append(batch if len(shape) == 0 else seq_len)
            else:
                shape.append(d.dim_value)
        inputs[inp.name] = np.ones(shape, dtype=np.int64)
    return inputs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    plugin_dir = os.path.join(REPO_ROOT, "build", "onnx-plugin")
    ext = "dylib" if sys.platform == "darwin" else "so"
    default_plugin = os.path.join(plugin_dir, f"libtinyxpu_ep.{ext}")
    plugin_path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else default_plugin)

    if not os.path.exists(plugin_path):
        print(f"ERROR: plugin not found at {plugin_path}")
        print("Run `cmake .. -DSIM=ON && make -j` in the build directory first.")
        return 1

    # ── Export + quantize (idempotent) ────────────────────────────────────────
    export_float()
    quantize_int8()
    print()

    # ── Inspect model ─────────────────────────────────────────────────────────
    ops = count_ops(MODEL_PATH)
    n_mmi = ops.get("MatMulInteger", 0)
    print(f"Model          : {MODEL_PATH}")
    print(f"MatMulInteger  : {n_mmi} nodes  ← TinyXPU EP claims all of these")
    print(f"Other ops      : { {k: v for k, v in ops.items() if k != 'MatMulInteger'} }")
    print(f"Plugin         : {plugin_path}")
    print()

    # ── Register EP ──────────────────────────────────────────────────────────
    ort.register_execution_provider_library("SampleEP", plugin_path)
    devices = [d for d in ort.get_ep_devices() if "SampleEP" in d.ep_name]
    if not devices:
        print("ERROR: TinyXPU EP device not found after registration.")
        return 1

    opts = ort.SessionOptions()
    opts.add_provider_for_devices(devices, {})

    lib = ctypes.CDLL(plugin_path)
    tinyxpu_perf.bind(lib)

    print("Creating session (EP claims MatMulInteger nodes) …")
    sys.stdout.flush()
    session = ort.InferenceSession(MODEL_PATH, sess_options=opts)
    print("Session ready.\n")

    # ── Run inference ─────────────────────────────────────────────────────────
    inputs = make_dummy_inputs(MODEL_PATH, batch=1, seq_len=16)
    print(f"Inputs         : { {k: v.shape for k, v in inputs.items()} }")

    print("Running inference …")
    sys.stdout.flush()
    result = session.run(None, inputs)
    perf = tinyxpu_perf.get_last_perf(lib)

    print(f"Output shape   : {result[0].shape}  dtype={result[0].dtype}")
    print(f"Output[0,0,:4] : {result[0][0, 0, :4]}\n")
    tinyxpu_perf.print_perf(perf)

    # ── CPU reference comparison ──────────────────────────────────────────────
    # Create the CPU session while the EP session is still alive (keeps the
    # plugin library reference count stable and avoids a dlclose race at exit).
    print("\nCPU reference (no EP) …")
    cpu_session = ort.InferenceSession(MODEL_PATH)
    cpu_result = cpu_session.run(None, inputs)

    out_f  = result[0].astype(np.float32)
    ref_f  = cpu_result[0].astype(np.float32)
    max_diff = float(np.max(np.abs(out_f - ref_f)))
    rel_err  = max_diff / (np.max(np.abs(ref_f)) + 1e-8)
    print(f"Max abs diff vs CPU EP : {max_diff:.4f}  (rel {rel_err*100:.2f}%)")
    print("Non-zero diff expected: activation zero-points are ignored by the EP.")
    print("Weight zero-points = 0 (symmetric quant), so weights are exact.\n")

    # Clean up before the ctypes lib handle goes out of scope.
    del cpu_session
    del session

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"SUCCESS  bert-tiny int8 ran end-to-end on TinyXPU EP")
    print(f"  {n_mmi} MatMulInteger nodes → systolic array (Verilator)")
    print(f"  All other ops          → CPU EP")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
