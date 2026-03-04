"""
Python-side mirror of TinyXpuPerfCounters / SimObservations from tinyxpu_perf.h.

Retrieves structured performance data from the TinyXPU EP plugin via ctypes
after an onnxruntime session.run() call.  Works for both Verilator simulation
and real hardware — the EP fills the same struct either way.

Usage:
    import ctypes
    import tinyxpu_perf

    lib = ctypes.CDLL(plugin_path)   # already loaded by ORT; same handle
    perf = tinyxpu_perf.get_last_perf(lib)
    tinyxpu_perf.print_perf(perf)
    print(perf.mac_efficiency)       # direct field access
"""
import ctypes
import dataclasses
from typing import Any


# ── ctypes struct layout ──────────────────────────────────────────────────────
# Must exactly match tinyxpu_perf.h.
# static constexpr members (kMacPJ, kSramPJ, kDramPJ) have no struct storage.

class _SimObservations(ctypes.Structure):
    _fields_ = [
        ("M",                  ctypes.c_int64),
        ("K",                  ctypes.c_int64),
        ("N",                  ctypes.c_int64),
        ("hw_rows",            ctypes.c_int),
        ("hw_cols",            ctypes.c_int),
        ("ticks_total",        ctypes.c_int64),
        ("ticks_reset",        ctypes.c_int64),
        ("ticks_weight_load",  ctypes.c_int64),
        ("ticks_streaming",    ctypes.c_int64),
        ("weight_writes",      ctypes.c_int64),
        ("activation_writes",  ctypes.c_int64),
        ("output_reads",       ctypes.c_int64),
    ]

class _TinyXpuPerfCounters(ctypes.Structure):
    _fields_ = [
        ("obs",               _SimObservations),
        ("hw_mac_events",     ctypes.c_int64),
        ("useful_mac_ops",    ctypes.c_int64),
        ("mac_efficiency",    ctypes.c_double),
        ("ticks_overhead",    ctypes.c_int64),
        ("overhead_frac",     ctypes.c_double),
        ("weight_bytes",      ctypes.c_int64),
        ("activation_bytes",  ctypes.c_int64),
        ("output_bytes",      ctypes.c_int64),
        ("total_mem_bytes",   ctypes.c_int64),
        ("ai_systolic",       ctypes.c_double),
        ("scalar_mem_bytes",  ctypes.c_int64),
        ("ai_scalar",         ctypes.c_double),
        # static constexpr kMacPJ / kSramPJ / kDramPJ — no storage
        ("e_compute_pj",      ctypes.c_double),
        ("e_weight_pj",       ctypes.c_double),
        ("e_activation_pj",   ctypes.c_double),
        ("e_output_pj",       ctypes.c_double),
        ("e_total_pj",        ctypes.c_double),
        ("e_scalar_pj",       ctypes.c_double),
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def bind(lib: ctypes.CDLL) -> None:
    """Set argtypes/restype on the exported getter (call once per loaded lib)."""
    lib.tinyxpu_get_last_perf.argtypes = [ctypes.POINTER(_TinyXpuPerfCounters)]
    lib.tinyxpu_get_last_perf.restype  = None


def get_last_perf(lib: ctypes.CDLL) -> _TinyXpuPerfCounters:
    """Return a copy of the perf counters from the most recent session.run()."""
    perf = _TinyXpuPerfCounters()
    lib.tinyxpu_get_last_perf(ctypes.byref(perf))
    return perf


def print_perf(perf: _TinyXpuPerfCounters) -> None:
    """Print the same table as TinyXpuPerfCounters::print() but from Python."""
    o = perf.obs
    print(f"\n[TinyXPU perf] MatMulInteger ({o.M}x{o.K}) x ({o.K}x{o.N})"
          f"  array {o.hw_rows}x{o.hw_cols}")

    print("  -- Clock cycles (observed) " + "-" * 40)
    print(f"  Total ticks         : {o.ticks_total}")
    print(f"    reset  (rst_n=0)  : {o.ticks_reset}")
    print(f"    weight load       : {o.ticks_weight_load}")
    print(f"    streaming (en=1)  : {o.ticks_streaming}")
    other = o.ticks_total - o.ticks_reset - o.ticks_weight_load - o.ticks_streaming
    print(f"    other overhead    : {other}")
    print(f"  Overhead fraction   : {perf.overhead_frac*100:.1f}%  (shrinks as M grows)")

    print("  -- Compute " + "-" * 56)
    print(f"  Physical MACs       : {perf.hw_mac_events}"
          f"  ({o.ticks_streaming} ticks x {o.hw_rows}x{o.hw_cols} PEs)")
    print(f"  Useful MACs         : {perf.useful_mac_ops}"
          f"  (M x K x N = {o.M}x{o.K}x{o.N})")
    pipeline_lat = o.hw_rows + o.hw_cols - 2
    print(f"  MAC efficiency      : {perf.mac_efficiency*100:.1f}%"
          f"  (= M/(M+pipeline_lat) = {o.M}/{o.M+pipeline_lat}, pipeline_lat={pipeline_lat})")

    print("  -- Data movement " + "-" * 50)
    print(f"  Weight writes       : {o.weight_writes} elems x 1 B = {perf.weight_bytes} B  (SRAM, once)")
    print(f"  Activation writes   : {o.activation_writes} elems x 1 B = {perf.activation_bytes} B  (DRAM, streamed)")
    print(f"  Output reads        : {o.output_reads} elems x 4 B = {perf.output_bytes} B  (DRAM, written)")
    print(f"  Total               : {perf.total_mem_bytes} B"
          f"  ->  arith. intensity {perf.ai_systolic:.2f} MAC/B")
    print(f"  Scalar no-cache     : {perf.scalar_mem_bytes} B"
          f"  ->  arith. intensity {perf.ai_scalar:.2f} MAC/B")
    print(f"  Weight reuse factor : {o.M}x"
          f"  (same {perf.weight_bytes} B loaded once, {o.M} output rows)")

    print("  -- Energy estimates (pJ) " + "-" * 42)
    print(f"  Constants: MAC=0.1 pJ  SRAM=2.0 pJ/B  DRAM=25.0 pJ/B")
    ratio = perf.hw_mac_events / perf.useful_mac_ops if perf.useful_mac_ops else 0
    print(f"  Compute (phys MACs) : {perf.e_compute_pj:7.1f} pJ  ({ratio:.1f}x more than useful-only)")
    print(f"  Weight load (SRAM)  : {perf.e_weight_pj:7.1f} pJ")
    print(f"  Activations (DRAM)  : {perf.e_activation_pj:7.1f} pJ")
    print(f"  Output (DRAM)       : {perf.e_output_pj:7.1f} pJ")
    print(f"  Systolic total      : {perf.e_total_pj:7.1f} pJ")
    scalar_ratio = perf.e_scalar_pj / perf.e_total_pj if perf.e_total_pj else 0
    print(f"  Scalar baseline     : {perf.e_scalar_pj:7.1f} pJ  ({scalar_ratio:.1f}x)")
