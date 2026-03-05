"""
Roofline plot for the TinyXPU systolic array.

Sweeps M (batch rows) and plots each operating point on a standard roofline
diagram: arithmetic intensity (MAC/B) vs. performance (MACs/cycle), against
the compute ceiling and memory-bandwidth rooflines.

Usage:
    pip install matplotlib
    python scripts/plot_roofline.py [path/to/libtinyxpu_ep.so]

Saves roofline.png in the scripts/ directory.
"""

import ctypes
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import tinyxpu_perf


def _collect(session, lib, rng, M_values):
    rows = []
    for m in M_values:
        A = rng.integers(-64, 64, size=(m, 4), dtype=np.int8)
        session.run(None, {"X": A})
        p = tinyxpu_perf.get_last_perf(lib)
        o = p.obs
        perf = p.useful_mac_ops / o.ticks_streaming if o.ticks_streaming else 0.0
        rows.append((m, p.ai_systolic, perf, o.hw_rows * o.hw_cols, p.ai_scalar))
    return rows  # [(M, ai_systolic, macs_per_cycle, peak_compute, ai_scalar), ...]


def plot(rows, out_path):
    peak = rows[0][3]  # MACs/cycle (constant across rows)
    ai_scalar = rows[0][4]  # constant across M (no weight reuse in scalar loop)

    fig, ax = plt.subplots(figsize=(8, 6))

    # ── Rooflines ──────────────────────────────────────────────────────────────
    bw = 16
    ridge_ai = peak / bw  # AI where bandwidth line meets compute ceiling
    ai_range = np.logspace(-2, 2, 1000)

    # Diagonal bandwidth line (memory-bound region only, up to ridge point)
    bw_ai = ai_range[ai_range <= ridge_ai]
    ax.loglog(bw_ai, bw * bw_ai, "-", color="#4CAF50", linewidth=2,
              label=f"BW = {bw} B/cyc (typical DRAM)")
    ax.plot(ridge_ai, peak, "o", color="#4CAF50", markersize=6, zorder=4)

    # Horizontal compute ceiling (compute-bound region only, from ridge point rightward)
    ax.plot([ridge_ai, 10], [peak, peak], "-", color="red", linewidth=2,
            label=f"Compute ceiling ({peak} MACs/cycle)")

    # ── Operating points ───────────────────────────────────────────────────────
    ais = [r[1] for r in rows]
    perfs = [r[2] for r in rows]
    ms = [r[0] for r in rows]

    ax.scatter(ais, perfs, color="black", zorder=5, s=45)
    for ai, perf, m in zip(ais, perfs, ms):
        ax.annotate(
            f"M={m}",
            (ai, perf),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=8,
            color="black",
        )

    # ── Scalar baseline AI ─────────────────────────────────────────────────────
    # ai_scalar is constant (no weight reuse): every A and B element is re-read
    # for every output, so AI = K / (2K + 4) ≈ 0.33 regardless of M.
    # Shown as a vertical line — no measured performance, just the X position.
    ax.axvline(
        ai_scalar, color="purple", linewidth=1.5, linestyle=":",
        label=f"Scalar no-cache AI = {ai_scalar:.2f} MAC/B",
    )

    # ── Labels ─────────────────────────────────────────────────────────────────
    ax.set_xlabel("Arithmetic Intensity (MACs / byte)", fontsize=12)
    ax.set_ylabel("Performance (MACs / cycle)", fontsize=12)
    ax.set_title("TinyXPU Roofline — 4×4 weight-stationary systolic array", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.5, peak * 4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(script_dir, "..")

    model_path = os.path.join(script_dir, "matmul_integer_4x4.onnx")
    default_so = os.path.join(repo_root, "build", "onnx-plugin", "libtinyxpu_ep.so")
    plugin_path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else default_so)

    for path, name in [(model_path, "model"), (plugin_path, "plugin")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}")
            return 1

    ort.register_execution_provider_library("SampleEP", plugin_path)
    devices = [d for d in ort.get_ep_devices() if "SampleEP" in d.ep_name]
    if not devices:
        print("ERROR: TinyXPU EP device not found")
        return 1

    opts = ort.SessionOptions()
    opts.add_provider_for_devices(devices, {})
    lib = ctypes.CDLL(plugin_path)
    tinyxpu_perf.bind(lib)

    sys.stdout.flush()
    session = ort.InferenceSession(model_path, sess_options=opts)

    rng = np.random.default_rng(42)
    M_values = [1, 2, 4, 8, 16, 32, 64]
    rows = _collect(session, lib, rng, M_values)

    del session
    ort.unregister_execution_provider_library("SampleEP")

    out_path = os.path.join(script_dir, "roofline.png")
    plot(rows, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
