#include "tinyxpu_perf.h"
#include <cstdio>

TinyXpuPerfCounters TinyXpuPerfCounters::from_observations(const SimObservations& obs)
{
    TinyXpuPerfCounters c{};
    c.obs = obs;

    // ── Compute ──────────────────────────────────────────────────────────────
    // Every en=1 tick, every PE fires.  This is the physical MAC count —
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
                  ? (double)c.hw_mac_events / c.total_mem_bytes : 0.0;

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

void TinyXpuPerfCounters::print() const
{
    const auto& o = obs;
    printf("\n[TinyXPU perf] MatMulInteger (%lldx%lld) × (%lldx%lld)  "
           "array %dx%d\n",
           (long long)o.M, (long long)o.K,
           (long long)o.K, (long long)o.N,
           o.hw_rows, o.hw_cols);

    printf("  ── Clock cycles (observed) ───────────────────────────────────────\n");
    printf("  Total ticks         : %lld\n",    (long long)obs.ticks_total);
    printf("    reset  (rst_n=0)  : %lld\n",    (long long)obs.ticks_reset);
    printf("    weight load       : %lld\n",    (long long)obs.ticks_weight_load);
    printf("    streaming (en=1)  : %lld\n",    (long long)obs.ticks_streaming);
    printf("    other overhead    : %lld\n",    (long long)(obs.ticks_total
                                                 - obs.ticks_reset
                                                 - obs.ticks_weight_load
                                                 - obs.ticks_streaming));
    printf("  Overhead fraction   : %.1f%%  (shrinks as M grows)\n",
           overhead_frac * 100.0);

    printf("  ── Compute (derived from ticks × PE count) ───────────────────────\n");
    printf("  Physical MACs       : %lld  (%lld streaming ticks × %d×%d PEs)\n",
           (long long)hw_mac_events, (long long)obs.ticks_streaming,
           obs.hw_rows, obs.hw_cols);
    printf("  Useful MACs         : %lld  (M×K×N = %lld×%lld×%lld)\n",
           (long long)useful_mac_ops,
           (long long)o.M, (long long)o.K, (long long)o.N);
    printf("  MAC efficiency      : %.1f%%  (= K/(hw_rows+hw_cols) = %lld/%d)\n",
           mac_efficiency * 100.0,
           (long long)o.K, obs.hw_rows + obs.hw_cols);
    printf("  [Pipeline note] The remaining %.1f%% of physical MACs are\n"
           "  pipeline fill/drain — real hardware energy, not useful compute.\n",
           (1.0 - mac_efficiency) * 100.0);

    printf("  ── Data movement (observed element counts → bytes) ───────────────\n");
    printf("  Weight writes       : %lld elems × 1 B = %lld B  (SRAM, loaded once)\n",
           (long long)obs.weight_writes,   (long long)weight_bytes);
    printf("  Activation writes   : %lld elems × 1 B = %lld B  (DRAM, streamed)\n",
           (long long)obs.activation_writes, (long long)activation_bytes);
    printf("  Output reads        : %lld elems × 4 B = %lld B  (DRAM, written)\n",
           (long long)obs.output_reads,    (long long)output_bytes);
    printf("  Total               : %lld B   →  arith. intensity %.2f MAC/B\n",
           (long long)total_mem_bytes, ai_systolic);
    printf("  Scalar no-cache     : %lld B   →  arith. intensity %.2f MAC/B\n",
           (long long)scalar_mem_bytes, ai_scalar);
    printf("  Weight reuse factor : %lld×  (same %lld B loaded once, %lld output rows)\n",
           (long long)o.M, (long long)weight_bytes, (long long)o.M);

    printf("  ── Energy estimates (pJ) ─────────────────────────────────────────\n");
    printf("  Constants: MAC=%.1fpJ  SRAM=%.1fpJ/B  DRAM=%.1fpJ/B\n",
           kMacPJ, kSramPJ, kDramPJ);
    printf("  Compute (phys MACs) : %7.1f pJ  (%.1fx more than useful-only)\n",
           e_compute_pj,
           (useful_mac_ops > 0) ? (double)hw_mac_events / useful_mac_ops : 0.0);
    printf("  Weight load (SRAM)  : %7.1f pJ\n", e_weight_pj);
    printf("  Activations (DRAM)  : %7.1f pJ\n", e_activation_pj);
    printf("  Output (DRAM)       : %7.1f pJ\n", e_output_pj);
    printf("  Systolic total      : %7.1f pJ\n", e_total_pj);
    printf("  Scalar baseline     : %7.1f pJ  (%.1fx)\n",
           e_scalar_pj,
           (e_total_pj > 0) ? e_scalar_pj / e_total_pj : 0.0);
    fflush(stdout);
}
