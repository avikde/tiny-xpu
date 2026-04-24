import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

# Must match the ROWS/COLS parameters the DUT is elaborated with.
# The runner at the bottom passes -Parray.ROWS=ROWS etc. to iverilog.
ROWS = 4
COLS = 4

async def reset_dut(dut):
    """Apply active-low reset for a few cycles."""
    dut.rst_n.value = 0
    for c in range(COLS):
        dut.weight_ld[c].value = 0
    for r in range(ROWS):
        dut.data_in_left[r].value = 0
    for c in range(COLS):
        dut.acc_in_top[c].value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_weights(dut, W):
    """Load weights via systolic cascade from top edge over ROWS cycles.
    This does not do any pipelining with the MACs since it is a test.

    At cycle t (0-indexed), row t of B is driven at the top and cascades down.
    After weight_ld=0, weights need ROWS more cycles to cascade to the bottom row.
    Returns when all PEs have their weights.
    """
    for c in range(COLS):
        dut.weight_ld[c].value = 1
    for load_row in range(W.shape[1]):
        for c in range(COLS):
            dut.acc_in_top[c].value = int(W[ROWS - load_row - 1][c]) # reversed
        await RisingEdge(dut.clk)
    for c in range(COLS):
        dut.weight_ld[c].value = 0
    # Can accept new inputs before the next clock cycle


async def stream_and_collect(dut, A, B = None):
    """Stream M rows of activations and collect M rows of outputs.
    Assumes weights are already loaded.

    Does input staggering
    At cycle t (1-indexed), data_in[r] carries A[t-r][r] if t >= r+1
    and t-r < M, else 0.  This places row r's first element r cycles
    after row 0's first element, matching the PE pipeline depth.

    No hardware de-skew: column j of output row i is valid at hardware cycle
    i + ROWS + j (1-indexed).  cocotb reads values after the NBA commit, so
    one additional edge is needed → readable after edge i + ROWS + j + 1.

    The loop runs for M + ROWS + COLS edges total, driving inputs for
    the first M edges and collecting each (row, col) pair as it becomes valid.

    The pictures here are helpful:
        https://www.avikde.me/p/systolic-arrays-for-general-robotics
    """
    M = len(A)
    results = np.zeros((M, COLS), dtype=np.int32)
    # Size of B should be M x COLS as well
    if B is not None:
        assert len(B) == M and all(len(row) == COLS for row in B)

    cycles_needed = M + ROWS + COLS
    for t in range(1, cycles_needed):
        # Rows enter as /x2/x1/ -> staggered
        for r in range(ROWS):
            src_row = t - r - 1  # A index for this row at this cycle
            if 0 <= src_row < M:
                dut.data_in_left[r].value = int(A[src_row][r])
            else:
                dut.data_in_left[r].value = 0
        # Biases enter as:
        # [    b22]
        # [b21 b12]
        # [b11  0 ]
        for c in range(COLS):
            dut.acc_in_top[c].value = 0
            src_row = t - c - 1  # A index for this row at this cycle
            if B is not None and 0 <= src_row < M:
                dut.acc_in_top[c].value = int(B[src_row][c])

        await RisingEdge(dut.clk)

        # After edge t, column j of row out_row = t - ROWS - j - 1 is readable.
        for j in range(COLS):
            out_row = t - ROWS - j - 1
            if 0 <= out_row < M:
                results[out_row][j] = dut.acc_out_bottom[j].value.to_signed()

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
        assert dut.data_out_right[r].value.to_signed() == 0, (
            f"data_out[{r}] not zero after reset"
        )
    for c in range(COLS):
        assert dut.acc_out_bottom[c].value.to_signed() == 0, (
            f"acc_out[{c}] not zero after reset"
        )


@cocotb.test()
async def test_weight_cascade_debug(dut):
    """Debug: Trace weight loading cascade step by step."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    W = np.random.randint(1, 6, size=(ROWS, COLS), dtype=np.int8)
    print("W = ", W)
    await load_weights(dut, W)

    # Set weight_ld = 1 and clock to see if we get the weights back
    for c in range(COLS):
        dut.weight_ld[c].value = 1

    # Read acc_out_bottom to verify weight cascade
    W2 = np.empty_like(W)
    for r in range(ROWS):
        await RisingEdge(dut.clk)
        for c in range(COLS):
            W2[ROWS - 1 - r, c] = dut.acc_out_bottom[c].value.to_signed()
    print("W2 = ", W2)
    assert np.array_equal(W2, W), f"acc_out_bottom=\n{W2}\n!=\nW=\n{W}"


@cocotb.test()
async def test_matmul_identity(dut):
    """Multiplying by the identity matrix should return the input unchanged."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    W = np.eye(ROWS, dtype=np.int8)
    await load_weights(dut, W)

    M = 1
    A = np.random.randint(1, 6, size=(M, ROWS), dtype=np.int8)

    results = await stream_and_collect(dut, A)

    assert np.array_equal(A, results), f"results=\n{results}\n!=\nA=\n{A}"
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

    M = 10
    # Small positive int8 values: products ≤ 5×5=25, column sum ≤ 4×25=100 → fits int32 easily
    X = np.random.randint(1, 6, size=(M, ROWS), dtype=np.int8)  # shape M×K
    W = np.random.randint(1, 6, size=(ROWS, COLS), dtype=np.int8)  # shape K×N
    B = np.random.randint(1, 6, size=(M, COLS), dtype=np.int8)  # shape M×N
    Y_expected = X.astype(np.int32) @ W.astype(np.int32) + B.astype(np.int32)  # shape M×N

    await load_weights(dut, W)
    Y = await stream_and_collect(dut, X, B)

    assert np.array_equal(Y, Y_expected), f"Y=\n{Y}\n!=\nY_expected=\n{Y_expected}"


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
