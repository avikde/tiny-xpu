import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

# Must match the ROWS/COLS parameters the DUT is elaborated with.
# The runner at the bottom passes -Parray.ROWS=ROWS etc. to iverilog.
ROWS = 4
COLS = 4

# Pipeline latency from the testbench's perspective: the number of clock edges
# to wait after presenting streaming cycle i before acc_out reflects row i.
# Hardware computes the result at tick i + ROWS + COLS - 1, but cocotb reads
# signal values before the current posedge's non-blocking assignments commit,
# so one additional edge is needed to observe the result → ROWS + COLS total.
PIPELINE_LAT = ROWS + COLS


async def reset_dut(dut):
    """Apply active-low reset for a few cycles."""
    dut.rst_n.value = 0
    dut.en.value = 0
    dut.weight_ld.value = 0
    for r in range(ROWS):
        dut.data_in[r].value = 0
        for c in range(COLS):
            dut.weight_in[r * COLS + c].value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_weights(dut, B):
    """Load a ROWS×COLS weight matrix (nested list or 2-D numpy array)."""
    dut.weight_ld.value = 1
    for r in range(ROWS):
        for c in range(COLS):
            dut.weight_in[r * COLS + c].value = int(B[r][c])
    await RisingEdge(dut.clk)
    dut.weight_ld.value = 0
    await RisingEdge(dut.clk)  # let weights settle


async def stream_and_collect(dut, A):
    """Stream M rows of activations and collect M rows of outputs.

    With hardware input skewing and output de-skewing, the driver simply
    presents row i of A on data_in[*] at streaming cycle i.  The module
    delays each row internally and aligns acc_out[*] so that all columns
    for output row i are valid simultaneously at cycle i + ROWS + COLS - 1.

    Total latency: M + max(0, ROWS + COLS - 1 - M) = ROWS + COLS - 1 cycles
    from the first drive until the first output is read.
    """
    M = len(A)

    # Stream all M input rows, one per clock cycle
    for i in range(M):
        for r in range(ROWS):
            dut.data_in[r].value = int(A[i][r])
        await RisingEdge(dut.clk)

    # Zero inputs after streaming
    for r in range(ROWS):
        dut.data_in[r].value = 0

    # We have already spent M clock edges.  Wait for the pipeline to produce
    # the first output (PIPELINE_LAT edges from cycle 0).
    extra_wait = PIPELINE_LAT - M
    if extra_wait > 0:
        await ClockCycles(dut.clk, extra_wait)

    # Collect M output rows; acc_out is stable immediately after each edge.
    results = []
    for i in range(M):
        results.append([dut.acc_out[j].value.to_signed() for j in range(COLS)])
        if i < M - 1:
            await RisingEdge(dut.clk)

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
        assert dut.data_out[r].value.to_signed() == 0, \
            f"data_out[{r}] not zero after reset"
    for c in range(COLS):
        assert dut.acc_out[c].value.to_signed() == 0, \
            f"acc_out[{c}] not zero after reset"


@cocotb.test()
async def test_matmul_identity(dut):
    """Multiplying by the identity matrix should return the input unchanged."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    I = [[1 if r == c else 0 for c in range(COLS)] for r in range(ROWS)]
    await load_weights(dut, I)

    A = [
        [1,  2,  3,  4],
        [5,  6,  7,  8],
        [9,  10, 11, 12],
        [13, 14, 15, 16],
    ]

    dut.en.value = 1
    results = await stream_and_collect(dut, A)

    for i, row in enumerate(A):
        for j in range(COLS):
            assert results[i][j] == row[j], (
                f"Identity check C[{i}][{j}]: expected {row[j]}, got {results[i][j]}"
            )


@cocotb.test()
async def test_matmul_4x4(dut):
    """Full 4×4 integer matrix multiply, verified against numpy.

    Computes C = A × B where A is M×K (M=4, K=ROWS) and B is K×N (N=COLS).
    data_in[k] carries the k-th element of each input row; the module
    applies input skewing internally.  acc_out[j] returns C[i][j] for all j
    simultaneously after the de-skew pipeline.
    """
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    rng = np.random.default_rng(42)
    # Small positive int8 values: products ≤ 5×5=25, column sum ≤ 4×25=100 → fits int32 easily
    A = rng.integers(1, 6, size=(4, ROWS), dtype=np.int8)   # shape M×K
    B = rng.integers(1, 6, size=(ROWS, COLS), dtype=np.int8)  # shape K×N
    C_expected = A.astype(np.int32) @ B.astype(np.int32)      # shape M×N

    await load_weights(dut, B)

    dut.en.value = 1
    results = await stream_and_collect(dut, A)

    for i in range(4):
        for j in range(COLS):
            expected = int(C_expected[i][j])
            assert results[i][j] == expected, (
                f"C[{i}][{j}]: expected {expected}, got {results[i][j]}"
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
