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
        rows.append((m, p.ai_systolic, perf, o.hw_rows * o.hw_cols))
    return rows  # [(M, ai, macs_per_cycle, peak_compute), ...]


def plot(rows, out_path):
    peak = rows[0][3]  # MACs/cycle (constant across rows)

    fig, ax = plt.subplots(figsize=(8, 6))

    # ── Rooflines ──────────────────────────────────────────────────────────────
    ai_range = np.logspace(-2, 2, 1000)
    bw_configs = [
        (4, "4 B/cyc (narrow DRAM)", "#2196F3"),
        (16, "16 B/cyc (typical DRAM)", "#4CAF50"),
        (64, "64 B/cyc (wide DRAM)", "#FF9800"),
    ]
    for bw, label, color in bw_configs:
        roof = np.minimum(peak, bw * ai_range)
        ax.loglog(
            ai_range, roof, "--", color=color, linewidth=1.5, label=f"BW = {label}"
        )
        # Ridge point: where bandwidth line meets compute ceiling
        ridge_ai = peak / bw
        ax.plot(ridge_ai, peak, "o", color=color, markersize=6, zorder=4)

    # Compute ceiling
    ax.axhline(
        peak, color="red", linewidth=2, label=f"Compute ceiling ({peak} MACs/cycle)"
    )

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

    # ── Labels ─────────────────────────────────────────────────────────────────
    ax.set_xlabel("Arithmetic Intensity (MACs / byte)", fontsize=12)
    ax.set_ylabel("Performance (MACs / cycle)", fontsize=12)
    ax.set_title("TinyXPU Roofline — 4×4 weight-stationary systolic array", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(0.1, 100)
    ax.set_ylim(0.5, peak * 4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    # plt.show()


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
    M_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    rows = _collect(session, lib, rng, M_values)

    del session
    ort.unregister_execution_provider_library("SampleEP")

    out_path = os.path.join(script_dir, "roofline.png")
    plot(rows, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
