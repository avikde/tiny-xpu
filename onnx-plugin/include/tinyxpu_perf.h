#pragma once
// TinyXPU simulation-grounded performance counters.
//
// Raw event counts (SimObservations) are populated by the Verilator driver
// loops in tinyxpu_ep.cpp — one counter increment per actual signal write,
// signal read, or clock tick.  Nothing here assumes knowledge of how a
// systolic array works; all derived quantities follow arithmetically from the
// observed counts, which can then be cross-checked against RTL expectations.
//
// Energy model (rough CMOS estimates; ratios more meaningful than absolutes):
//   int8 MAC           ~0.1  pJ
//   on-chip SRAM byte  ~2.0  pJ  (read or write)
//   off-chip DRAM byte ~25.0 pJ  (LPDDR4/DDR5)

#include <cstdint>

// ── Raw event counts from the Verilator driver ───────────────────────────────
// Every field is incremented exactly once per physical event in the driver.
struct SimObservations {
    int64_t M, K, N;            // problem size passed by the caller
    int     hw_rows, hw_cols;   // physical PE array size

    // Clock ticks (one increment per tick() call in the driver)
    int64_t ticks_total;
    int64_t ticks_reset;        // ticks while rst_n == 0
    int64_t ticks_weight_load;  // ticks while weight_ld == 1  (latch + settle)
    int64_t ticks_streaming;    // ticks while en == 1

    // Element-level data movement (one increment per element written/read)
    int64_t weight_writes;      // host → weight_in[]  (int8 elements)
    int64_t activation_writes;  // host → data_in[]    (int8 elements, all rows)
    int64_t output_reads;       // acc_out[]  → host   (int32 elements, all rows)
};

// ── Quantities derived from SimObservations ───────────────────────────────────
struct TinyXpuPerfCounters {
    SimObservations obs;        // the raw numbers, for reference

    // ── Compute ──────────────────────────────────────────────────────────────
    // Physical MAC operations: every tick where en=1, every PE fires once.
    // This is what the silicon actually spends energy on.
    int64_t hw_mac_events;      // = ticks_streaming * hw_rows * hw_cols

    // Minimum MACs needed for correct output (inner-product definition).
    int64_t useful_mac_ops;     // = M * K * N

    // Fraction of physical MACs that contribute a distinct output term.
    // = useful / hw  =  K / (hw_rows + hw_cols)  for this dataflow.
    // Captures pipeline fill/drain overhead; independent of M.
    double mac_efficiency;

    // ── Cycles ───────────────────────────────────────────────────────────────
    int64_t ticks_overhead;     // ticks_total - ticks_streaming
    double  overhead_frac;      // ticks_overhead / ticks_total  (shrinks with M)

    // ── Memory traffic (bytes, from observation counts × element widths) ─────
    int64_t weight_bytes;       // weight_writes  × 1  (int8)
    int64_t activation_bytes;   // activation_writes × 1  (int8)
    int64_t output_bytes;       // output_reads  × 4  (int32)
    int64_t total_mem_bytes;

    // Arithmetic intensity: physical MACs per memory byte transferred.
    // Roofline x-axis position.
    double ai_systolic;

    // Scalar triple-loop, no cache — worst-case analytical baseline.
    int64_t scalar_mem_bytes;   // 2×M×K×N×1 (A+B re-read) + M×N×4 (C written)
    double  ai_scalar;

    // ── Energy estimates (picojoules) ────────────────────────────────────────
    // Weights are assumed on-chip SRAM; activations and outputs off-chip DRAM.
    // hw_mac_events (not useful_mac_ops) drives compute energy: the hardware
    // pays for every physical MAC, not just the useful ones.
    static constexpr double kMacPJ  =  0.1;
    static constexpr double kSramPJ =  2.0;
    static constexpr double kDramPJ = 25.0;

    double e_compute_pj;        // hw_mac_events * kMacPJ
    double e_weight_pj;         // weight_bytes  * kSramPJ   (on-chip)
    double e_activation_pj;     // activation_bytes * kDramPJ
    double e_output_pj;         // output_bytes  * kDramPJ
    double e_total_pj;

    double e_scalar_pj;         // analytical scalar baseline

    // ── Factory and display ──────────────────────────────────────────────────
    static TinyXpuPerfCounters from_observations(const SimObservations& obs);
};
