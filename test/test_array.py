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


async def stream_row(dut, row):
    """Drive one row of int8 activations; wait for the pipeline to drain.

    Latency analysis (all outputs registered):
      - data_wire[k][j] sees data_in[k] after j east-pipeline stages  → j cycles
      - acc_wire[ROWS][j] accumulates spatially through ROWS south stages → ROWS more cycles
      - acc_out[j] is valid at edge  j + ROWS - 1  after data is applied

    Worst case (j = COLS-1):  COLS-1 + ROWS - 1 = ROWS + COLS - 2  cycles.
    Waiting ROWS + COLS edges gives one cycle of margin (mirrors test_pe.py's
    double-await pattern) and keeps all acc_out[0..COLS-1] stable simultaneously.
    """
    for k in range(ROWS):
        dut.data_in[k].value = int(row[k])
    await ClockCycles(dut.clk, ROWS + COLS)
    return [dut.acc_out[j].value.to_signed() for j in range(COLS)]


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

    # B = I
    I = [[1 if r == c else 0 for c in range(COLS)] for r in range(ROWS)]
    await load_weights(dut, I)

    A = [
        [1,  2,  3,  4],
        [5,  6,  7,  8],
        [9,  10, 11, 12],
        [13, 14, 15, 16],
    ]

    dut.en.value = 1
    for i, row in enumerate(A):
        result = await stream_row(dut, row)
        for j in range(COLS):
            assert result[j] == row[j], (
                f"Identity check C[{i}][{j}]: expected {row[j]}, got {result[j]}"
            )


@cocotb.test()
async def test_matmul_4x4(dut):
    """Full 4×4 integer matrix multiply, verified against numpy.

    Computes  C = A × B  where A is M×K (M=4, K=ROWS) and B is K×N (N=COLS).
    data_in[k] carries row k of the inner dimension for each output row i;
    acc_out[j] returns C[i][j] = Σ_k A[i][k] × B[k][j] after the pipeline drains.
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
    for i in range(4):  # M = 4 output rows
        result = await stream_row(dut, A[i])
        for j in range(COLS):
            expected = int(C_expected[i][j])
            assert result[j] == expected, (
                f"C[{i}][{j}]: expected {expected}, got {result[j]}"
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
