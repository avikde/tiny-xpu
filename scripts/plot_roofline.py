"""
Roofline plot for the TinyXPU systolic array.

Sweeps M (batch rows) across three weight shapes (4×16, 8×8, 16×4) and plots
each operating point on a standard roofline diagram: arithmetic intensity
(MAC/B) vs. performance (MACs/cycle), against the compute ceiling and
memory-bandwidth roofline.

All three weight shapes have 64 elements (K×N = 64), so they share the same
SRAM weight-load cost. The AI differences arise from K:N ratio effects on
activation reads (M×K bytes) and output writes (M×N×4 bytes).

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


def _collect(session, lib, rng, M_values, K):
    rows = []
    for m in M_values:
        A = rng.integers(-64, 64, size=(m, K), dtype=np.int8)
        session.run(None, {"X": A})
        p = tinyxpu_perf.get_last_perf(lib)
        o = p.obs
        perf = p.useful_mac_ops / o.ticks_streaming if o.ticks_streaming else 0.0
        latency = o.hw_rows + o.N - 2  # cycles to first output
        rows.append((m, p.ai_systolic, perf, o.hw_rows * o.hw_cols, latency))
    return rows  # [(M, ai_systolic, macs_per_cycle, peak_compute, latency), ...]


def plot(series, out_path):
    """
    series: list of (label, color, rows) where rows is from _collect().
    """
    peak = series[0][2][0][3]  # hw_rows * hw_cols — same for all series

    fig, ((ax_util, ax_lat), (ax, ax_empty)) = plt.subplots(2, 2, figsize=(10, 8))
    ax_empty.axis("off")
    markers = ["o", "s", "^", "D"]

    # ── Roofline ───────────────────────────────────────────────────────────────
    bw = 16
    x_max = 10

    # Two compute ceilings: the actual 16×16 array and a smaller 8×8 reference.
    peak_lo = 64  # 8×8 reference ceiling
    ridge_lo = peak_lo / bw  # = 4.0  (visible on plot)
    ridge_hi = peak / bw  # = 16.0 (off-screen)

    # BW diagonals
    bw_hi = 64
    # 64 B/cyc line: thick solid gray (matches horizontal compute ceiling)
    ridge_bw_hi = peak / bw_hi  # = 4.0
    ax.loglog(
        [0.4, ridge_bw_hi],
        [bw_hi * 0.4, bw_hi * ridge_bw_hi],
        "-",
        color="gray",
        linewidth=2,
        label=f"Mem {bw_hi} B/cyc",
    )

    # 16 B/cyc line: dashed gray (matches lower dashed ceiling)
    ax.loglog(
        [0.4, x_max],
        [bw * 0.4, bw * x_max],
        "--",
        color="gray",
        linewidth=1.5,
        label=f"Mem {bw} B/cyc",
    )

    # Upper (solid) ceiling at peak MACs/cycle — ridge off-screen
    ax.axhline(peak, color="black", linewidth=2, label=f"{peak} MACs/cycle (16×16)")

    # Lower (dashed) ceiling at 64 MACs/cycle — ridge is visible at AI=4
    ax.axhline(
        peak_lo,
        color="black",
        linewidth=1.5,
        linestyle="--",
        label=f"{peak_lo} MACs/cycle (8×8 ref)",
    )

    # ── Operating points per weight shape ──────────────────────────────────────
    for (label, color, rows), marker in zip(series, markers):
        ais = [r[1] for r in rows]
        perfs = [r[2] for r in rows]
        ms = [r[0] for r in rows]

        ax.plot(ais, perfs, "-", color=color, linewidth=1.2, alpha=0.6)
        ax.scatter(
            ais,
            perfs,
            color=color,
            marker=marker,
            zorder=5,
            s=50,
            label=f"W shape {label}",
        )
        # Annotate only the first and last M to avoid clutter
        for idx in (0, -1):
            ax.annotate(
                f"M={ms[idx]}",
                (ais[idx], perfs[idx]),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=7,
                color=color,
            )

    # ── Labels ─────────────────────────────────────────────────────────────────
    ax.set_xlabel("Arithmetic Intensity (MACs / byte)", fontsize=12)
    ax.set_ylabel("Performance (MACs / cycle)", fontsize=12)
    ax.set_title(
        "TinyXPU Roofline 16x16-PE weight-stationary systolic array", fontsize=12
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(0.4, x_max)
    ax.set_ylim(0.5, peak * 4)

    # ── PE utilization ─────────────────────────────────────────────────────────
    # Measured efficiency from each weight-shape series.
    for (label, color, rows), marker in zip(series, markers):
        ms = [r[0] for r in rows]
        eff = [r[2] / r[3] * 100 for r in rows]  # macs_per_cycle / peak * 100
        ax_util.semilogx(ms, eff, "-", color=color, linewidth=1.2, alpha=0.6)
        ax_util.scatter(
            ms,
            eff,
            color=color,
            marker=marker,
            zorder=5,
            s=50,
            label=f"W shape {label}",
        )
        # Mirror the roofline M annotations so points can be cross-referenced.
        for idx in (0, -1):
            ax_util.annotate(
                f"M={ms[idx]}",
                (ms[idx], eff[idx]),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=7,
                color=color,
            )

    ax_util.set_xlabel("M (batch rows)", fontsize=12)
    ax_util.set_ylabel("PE utilization (%)", fontsize=12)
    ax_util.set_title("PE utilization  =  M / (M + pipeline_latency)", fontsize=12)
    ax_util.legend(fontsize=9, loc="upper left")
    ax_util.grid(True, which="both", alpha=0.25)
    ax_util.set_ylim(0, 105)
    ax_util.set_xlim(1, max(r[0] for r in series[0][2]))

    # ── Throughput vs Latency (analytical, fixed ROWS=16, varying N per K) ──
    # Array is 16×16.  Weight matrix is K×N with K ≤ 16, N ≤ 16.
    # Useful MACs per run = M * K * N.
    # Streaming ticks = M + HW_ROWS + N - 2.
    # Latency to first output = HW_ROWS + N - 2.
    # Throughput = M * K * N / (M + HW_ROWS + N - 2).
    M_fixed = 256
    HW_R = 16
    k_configs = [
        (4,  "#4CAF50"),
        (8,  "#FF9800"),
        (16, "#2196F3"),
    ]
    total_size = 64  # K * N_max = constant across lines

    for K, color in k_configs:
        N_values = np.arange(1, total_size // K + 1)
        lats = HW_R + N_values - 2
        throughputs = M_fixed * K * N_values / (M_fixed + lats)
        ax_lat.plot(lats, throughputs, "-o", color=color, markersize=3,
                    linewidth=1.2, label=f"K={K}")
        for idx in (0, -1):
            ax_lat.annotate(
                f"N={N_values[idx]}",
                (lats[idx], throughputs[idx]),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=7,
                color=color,
            )

    ax_lat.axhline(64, color="black", linewidth=1.5, linestyle="--",
                    label="64 MACs/cycle")
    ax_lat.set_xlabel("Latency (cycles to first output = ROWS + N − 2)", fontsize=12)
    ax_lat.set_ylabel("Throughput (MACs / cycle)", fontsize=12)
    ax_lat.set_title(f"Throughput vs. latency (ROWS=16, M={M_fixed})", fontsize=12)
    ax_lat.legend(fontsize=9, loc="lower right")
    ax_lat.set_yscale("log")
    ax_lat.grid(True, which="both", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    import matplotlib

    if matplotlib.get_backend() != "agg":
        plt.show()


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(script_dir, "..")

    plugin_lib = (
        "libtinyxpu_ep.dylib" if sys.platform == "darwin" else "libtinyxpu_ep.so"
    )
    default_so = os.path.join(repo_root, "build", "onnx-plugin", plugin_lib)
    plugin_path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else default_so)

    if not os.path.exists(plugin_path):
        print(f"ERROR: plugin not found: {plugin_path}")
        return 1

    # (display label, plot color, model filename, K = cols of A = rows of B)
    model_configs = [
        ("4×16", "#2196F3", "matmul_integer_4x16.onnx", 4),
        ("8×8", "#FF9800", "matmul_integer_8x8.onnx", 8),
        ("16×4", "#9C27B0", "matmul_integer_16x4.onnx", 16),
        ("16×16", "#E53935", "matmul_integer_16x16.onnx", 16),
    ]

    ort.register_execution_provider_library("SampleEP", plugin_path)
    devices = [d for d in ort.get_ep_devices() if "SampleEP" in d.ep_name]
    if not devices:
        print("ERROR: TinyXPU EP device not found")
        return 1

    opts = ort.SessionOptions()
    opts.add_provider_for_devices(devices, {})
    lib = ctypes.CDLL(plugin_path)
    tinyxpu_perf.bind(lib)

    rng = np.random.default_rng(42)
    M_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    series = []
    for label, color, model_file, K in model_configs:
        model_path = os.path.join(script_dir, model_file)
        if not os.path.exists(model_path):
            print(f"ERROR: model not found: {model_path}")
            return 1

        sys.stdout.flush()
        session = ort.InferenceSession(model_path, sess_options=opts)
        rows = _collect(session, lib, rng, M_values, K)
        del session
        series.append((label, color, rows))

    ort.unregister_execution_provider_library("SampleEP")

    out_path = os.path.join(script_dir, "roofline.png")
    plot(series, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
