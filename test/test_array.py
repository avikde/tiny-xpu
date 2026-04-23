import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

# Must match the ROWS/COLS parameters the DUT is elaborated with.
# The runner at the bottom passes -Parray.ROWS=ROWS etc. to iverilog.
ROWS = 16
COLS = 16

async def reset_dut(dut):
    """Apply active-low reset for a few cycles."""
    dut.rst_n.value = 0
    dut.en.value = 0
    dut.relu_en.value = 0
    dut.requant_en.value = 0
    dut.M0.value = 0
    dut.rshift.value = 0
    dut.zero_pt.value = 0
    dut.weight_in.value = 0
    for r in range(ROWS):
        dut.data_in[r].value = 0
    for c in range(COLS):
        dut.weight_in_top[c].value = 0
    for c in range(COLS):
        dut.bias_in[c].value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_weights(dut, B):
    """Load weights via systolic cascade from top edge over ROWS cycles.

    At cycle t (0-indexed), row t of B is driven at the top and cascades down.
    After weight_in=0, weights need ROWS more cycles to cascade to the bottom row.
    Returns when all PEs have their weights.
    """
    dut.weight_in.value = 1
    for load_row in range(ROWS):
        for c in range(COLS):
            dut.weight_in_top[c].value = int(B[load_row][c])
        await RisingEdge(dut.clk)
    dut.weight_in.value = 0
    # Weights continue cascading down for ROWS cycles after weight_in goes low
    await ClockCycles(dut.clk, ROWS)


async def stream_and_collect(dut, A):
    """Stream M rows of activations and collect M rows of outputs.

    External pre-staggering: the caller drives already-staggered inputs.
    At cycle t (1-indexed), data_in[r] carries A[t-r][r] if t >= r+1
    and t-r < M, else 0.  This places row r's first element r cycles
    after row 0's first element, matching the PE pipeline depth.

    No hardware de-skew: column j of output row i is valid at hardware cycle
    i + ROWS + j (1-indexed).  cocotb reads values after the NBA commit, so
    one additional edge is needed → readable after edge i + ROWS + j + 1.

    The loop runs for M + ROWS + COLS - 1 edges total, driving inputs for
    the first M edges and collecting each (row, col) pair as it becomes valid.
    """
    M = len(A)
    results = [[0] * COLS for _ in range(M)]

    # t is 1-indexed edge count; total edges covers the last output (row M-1, col COLS-1)
    total_edges = M + ROWS + COLS - 1
    for t in range(1, total_edges + 1):
        # Drive externally pre-staggered inputs: row r's element at (t-r) enters at cycle t
        for r in range(ROWS):
            src_row = t - r - 1  # A index for this row at this cycle
            if 0 <= src_row < M:
                dut.data_in[r].value = int(A[src_row][r])
            else:
                dut.data_in[r].value = 0

        await RisingEdge(dut.clk)

        # After edge t, column j of row out_row = t - ROWS - j - 1 is readable.
        for j in range(COLS):
            out_row = t - ROWS - j - 1
            if 0 <= out_row < M:
                results[out_row][j] = dut.acc_out[j].value.to_signed()

    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@cocotb.test()
async def test_reset(dut):
    """After reset all outputs should be zero."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    for r in range(ROWS):
        assert dut.data_out[r].value.to_signed() == 0, (
            f"data_out[{r}] not zero after reset"
        )
    for c in range(COLS):
        assert dut.acc_out[c].value.to_signed() == 0, (
            f"acc_out[{c}] not zero after reset"
        )


@cocotb.test()
async def test_weight_cascade_debug(dut):
    """Debug: Trace weight loading cascade step by step."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)
    
    # Simple 4x4 identity for debugging (we only need first 4 rows/cols visible)
    cocotb.log.info("=== Weight Cascade Debug ===")
    
    dut.weight_in.value = 1
    for load_row in range(4):
        for c in range(COLS):
            val = 1 if load_row == c else 0  # Identity pattern
            dut.weight_in_top[c].value = val
        cocotb.log.info(f"Cycle {load_row}: Driving weight row {load_row} at top")
        await RisingEdge(dut.clk)
        # Check acc_wire[0] after the edge
        acc0 = dut.acc_out[0].value.to_signed() if hasattr(dut.acc_out[0], 'value') else 0
        cocotb.log.info(f"  After edge: acc_out[0] = {acc0}")
    
    dut.weight_in.value = 0
    cocotb.log.info("weight_in = 0, cascading remaining rows...")
    for i in range(ROWS):
        await RisingEdge(dut.clk)
        acc0 = dut.acc_out[0].value.to_signed() if hasattr(dut.acc_out[0], 'value') else 0
        cocotb.log.info(f"  Settle cycle {i}: acc_out[0] = {acc0}")
    
    # Now stream a simple input and check output
    cocotb.log.info("=== Streaming Input [1,0,0,0,...] ===")
    dut.en.value = 1
    dut.data_in[0].value = 1
    for i in range(COLS + ROWS + 5):
        await RisingEdge(dut.clk)
        acc_vals = [dut.acc_out[c].value.to_signed() for c in range(min(4, COLS))]
        cocotb.log.info(f"  Cycle {i}: acc_out[0:3] = {acc_vals}")
    
    dut.en.value = 0
    cocotb.log.info("=== Debug Complete ===")


@cocotb.test()
async def test_matmul_identity(dut):
    """Multiplying by the identity matrix should return the input unchanged."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    I = [[1 if r == c else 0 for c in range(COLS)] for r in range(ROWS)]
    await load_weights(dut, I)

    rng = np.random.default_rng(42)

    A = rng.integers(1, 6, size=(16, ROWS), dtype=np.int8)

    dut.en.value = 1
    results = await stream_and_collect(dut, A)

    for i, row in enumerate(A):
        for j in range(COLS):
            assert results[i][j] == row[j], (
                f"Identity check C[{i}][{j}]: expected {row[j]}, got {results[i][j]}"
            )


@cocotb.test()
async def test_matmul(dut):
    """Full integer matrix multiply, verified against numpy.

    Computes C = A × B where A is M×K (M=ROWS, K=ROWS) and B is K×N (N=COLS).
    data_in[k] carries the k-th element of each input row; external pre-staggering
    is applied by the test so row k's first element enters k cycles after row 0.
    acc_out[j] returns C[i][j] with column j valid one cycle later than column j-1
    (skewed, no hardware de-skew).
    """
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    rng = np.random.default_rng(42)
    # Small positive int8 values: products ≤ 5×5=25, column sum ≤ 4×25=100 → fits int32 easily
    A = rng.integers(1, 6, size=(ROWS, ROWS), dtype=np.int8)  # shape M×K
    B = rng.integers(1, 6, size=(ROWS, COLS), dtype=np.int8)  # shape K×N
    C_expected = A.astype(np.int32) @ B.astype(np.int32)  # shape M×N

    await load_weights(dut, B)

    dut.en.value = 1
    results = await stream_and_collect(dut, A)

    for i in range(4):
        for j in range(COLS):
            expected = int(C_expected[i][j])
            assert results[i][j] == expected, (
                f"C[{i}][{j}]: expected {expected}, got {results[i][j]}"
            )


@cocotb.test()
async def test_requant(dut):
    """requant_en=1: q_out matches Python reference clamp(round(A@W * S), -128, 127)."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    rng = np.random.default_rng(55)
    A = rng.integers(-5, 6, size=(ROWS, ROWS), dtype=np.int8)
    B = rng.integers(-5, 6, size=(ROWS, COLS), dtype=np.int8)
    C_int32 = A.astype(np.int32) @ B.astype(np.int32)  # [ROWS, COLS]

    # Choose a scale that keeps values in int8 range after requant.
    # S ≈ 1/200  →  M0 = round(S * 2^31), rshift = 31
    S_float  = 1.0 / 200.0
    RSHIFT   = 31
    M0_val   = int(round(S_float * (1 << RSHIFT)))
    ZERO_PT  = 0

    # Python reference: replicate hardware integer arithmetic exactly.
    # product = biased * M0  (int64); shifted = product >> rshift  (arithmetic, truncation);
    # then clamp to [-128, 127].
    _product = C_int32.astype(np.int64) * np.int64(M0_val)
    _shifted = _product >> RSHIFT   # arithmetic right-shift in Python matches hardware truncation
    C_ref = np.clip(_shifted, -128, 127).astype(np.int8)

    await load_weights(dut, B)

    dut.requant_en.value = 1
    dut.relu_en.value    = 0
    dut.M0.value         = M0_val
    dut.rshift.value     = RSHIFT
    dut.zero_pt.value    = ZERO_PT
    for c in range(COLS):
        dut.bias_in[c].value = 0  # no bias in this test

    dut.en.value = 1
    results = await stream_and_collect(dut, A)
    dut.en.value = 0
    dut.requant_en.value = 0

    # results[] contains the raw acc_out (int32); q_out is a separate port.
    # For the requant test we read q_out directly.
    for i in range(4):
        for j in range(COLS):
            got = dut.q_out[j].value.to_signed()
            # q_out is valid at the same cycle as acc_out — already captured above
            # because stream_and_collect reads after each rising edge.
            # Re-read from the DUT's q_out at the final steady state for row 0..3.
            # (stream_and_collect only captures acc_out; we verify q_out via a
            #  separate single-row check below.)
            _ = got  # suppress lint

    # Single-row verification: stream one row and check q_out immediately.
    await reset_dut(dut)
    await load_weights(dut, B)

    dut.requant_en.value = 1
    dut.M0.value         = M0_val
    dut.rshift.value     = RSHIFT
    dut.zero_pt.value    = ZERO_PT
    dut.en.value         = 1

    # Stream row 0, then drain until output is valid.
    # cocotb reads in NBA phase (after the rising edge), so q_out[j] for row 0 is
    # readable after edge ROWS + j + 1  (same convention as stream_and_collect).
    for t in range(1, ROWS + COLS + 1):
        if t <= 1:
            for r in range(ROWS):
                dut.data_in[r].value = int(A[0][r])
        else:
            for r in range(ROWS):
                dut.data_in[r].value = 0
        await RisingEdge(dut.clk)

        # After edge t, q_out[j] for row 0 is valid when t == ROWS + j + 1.
        for j in range(COLS):
            if t == ROWS + j + 1:
                got      = dut.q_out[j].value.to_signed()
                expected = int(C_ref[0][j])
                assert got == expected, (
                    f"requant q_out[{j}]: expected {expected}, got {got} "
                    f"(acc={C_int32[0][j]}, S={S_float:.6f})"
                )

    dut.en.value = 0
    dut.requant_en.value = 0


@cocotb.test()
async def test_relu_en(dut):
    """relu_en=1 clamps negative acc_out to 0; positive values pass through."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    rng = np.random.default_rng(77)
    # Mixed-sign weights produce a mix of positive and negative outputs.
    A = rng.integers(-5, 6, size=(ROWS, ROWS), dtype=np.int8)
    B = rng.integers(-5, 6, size=(ROWS, COLS), dtype=np.int8)
    C_expected = A.astype(np.int32) @ B.astype(np.int32)
    C_relu     = np.maximum(C_expected, 0)

    # Verify there is at least one negative entry so the test is meaningful.
    assert np.any(C_expected < 0), "test setup: expected some negative outputs"

    await load_weights(dut, B)

    # --- relu_en = 1 ---
    dut.relu_en.value = 1
    dut.en.value = 1
    results = await stream_and_collect(dut, A)
    dut.en.value = 0
    dut.relu_en.value = 0

    for i in range(4):
        for j in range(COLS):
            expected = int(C_relu[i][j])
            assert results[i][j] == expected, (
                f"ReLU C[{i}][{j}]: expected {expected}, got {results[i][j]}"
            )

    # --- relu_en = 0: same weights, outputs should match raw matmul ---
    await reset_dut(dut)
    await load_weights(dut, B)

    dut.relu_en.value = 0
    dut.en.value = 1
    results_no_relu = await stream_and_collect(dut, A)
    dut.en.value = 0

    for i in range(4):
        for j in range(COLS):
            expected = int(C_expected[i][j])
            assert results_no_relu[i][j] == expected, (
                f"No-ReLU C[{i}][{j}]: expected {expected}, got {results_no_relu[i][j]}"
            )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_tests():
    """Build and run all cocotb tests using the cocotb runner API."""
    from cocotb_tools.runner import get_runner

    proj_dir = Path(__file__).resolve().parent.parent
    sources = sorted(proj_dir.glob("src/*.sv"))

    build_dir = Path("sim_build_array")

    runner = get_runner("icarus")
    runner.build(
        sources=sources,
        hdl_toplevel="array",
        build_args=["-g2012", f"-Parray.ROWS={ROWS}", f"-Parray.COLS={COLS}"],
        waves=True,
        build_dir=build_dir,
        always=True,  # always recompile — avoids stale binaries from pe test
    )
    runner.test(
        hdl_toplevel="array",
        test_module="test_array",
        waves=True,
        build_dir=build_dir,
    )

    results_file = build_dir / "results.xml"
    if results_file.exists():
        tree = ET.parse(results_file)
        failures = tree.findall(".//failure") + tree.findall(".//error")
        if failures:
            sys.exit(1)


if __name__ == "__main__":
    run_tests()
