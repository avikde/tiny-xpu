#include "tinyxpu_perf.h"

// Perf counters from the most recent MatMulInteger execution.
// Written by the driver; read by tinyxpu_get_last_perf().
// Not thread-safe: single-threaded Verilator simulation assumed.
static TinyXpuPerfCounters g_last_perf{};

void tinyxpu_set_last_perf(const TinyXpuPerfCounters& counters) {
    g_last_perf = counters;
}

const TinyXpuPerfCounters& tinyxpu_get_last_perf_ref() {
    return g_last_perf;
}

TinyXpuPerfCounters TinyXpuPerfCounters::from_observations(const SimObservations& obs)
{
    TinyXpuPerfCounters c{};
    c.obs = obs;

    // ── Compute ──────────────────────────────────────────────────────────────
    // Every streaming tick, every PE fires.  This is the physical MAC count —
    // derived purely from what the driver observed, not from knowing the
    // array topology.
    c.hw_mac_events  = obs.ticks_streaming * obs.hw_rows * obs.hw_cols;
    c.useful_mac_ops = obs.M * obs.K * obs.N;   // minimum for correct output
    c.mac_efficiency = (c.hw_mac_events > 0)
                     ? (double)c.useful_mac_ops / c.hw_mac_events : 0.0;

    // ── Cycles ───────────────────────────────────────────────────────────────
    c.ticks_overhead = obs.ticks_total - obs.ticks_streaming;
    c.overhead_frac  = (obs.ticks_total > 0)
                     ? (double)c.ticks_overhead / obs.ticks_total : 0.0;

    // ── Memory traffic ───────────────────────────────────────────────────────
    // Byte counts follow directly from element counts × data widths.
    // The driver counted every write/read; we just apply the sizes.
    c.weight_bytes     = obs.weight_writes     * 1;  // int8
    c.activation_bytes = obs.activation_writes * 1;  // int8
    c.output_bytes     = obs.output_reads      * 4;  // int32
    c.total_mem_bytes  = c.weight_bytes + c.activation_bytes + c.output_bytes;

    c.ai_systolic = (c.total_mem_bytes > 0)
                  ? (double)c.useful_mac_ops / c.total_mem_bytes : 0.0;

    // Scalar no-cache baseline (analytical — nothing to observe for a loop
    // that doesn't exist, but useful for the comparison).
    c.scalar_mem_bytes = obs.M * obs.K * obs.N * 1   // A: each elem read N×
                       + obs.M * obs.K * obs.N * 1   // B: each elem read M×
                       + obs.M * obs.N * 4;           // C: written once
    c.ai_scalar = (c.scalar_mem_bytes > 0)
                ? (double)c.useful_mac_ops / c.scalar_mem_bytes : 0.0;

    // ── Energy ───────────────────────────────────────────────────────────────
    // hw_mac_events drives compute energy: the PE array pays for every physical
    // multiply-accumulate, regardless of whether it contributes a new output.
    c.e_compute_pj    = c.hw_mac_events    * kMacPJ;
    c.e_weight_pj     = c.weight_bytes     * kSramPJ;  // weights stay on-chip
    c.e_activation_pj = c.activation_bytes * kDramPJ;
    c.e_output_pj     = c.output_bytes     * kDramPJ;
    c.e_total_pj      = c.e_compute_pj + c.e_weight_pj
                      + c.e_activation_pj + c.e_output_pj;

    c.e_scalar_pj = c.useful_mac_ops * kMacPJ   // same compute cost
                  + c.scalar_mem_bytes * kDramPJ; // all traffic off-chip

    return c;
}
