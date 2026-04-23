import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

ROWS = 16


async def reset_dut(dut):
    """Apply active-low reset for a few cycles."""
    dut.rst_n.value = 0
    dut.weight_ld.value = 0
    dut.acc_in_top.value = 0
    for r in range(ROWS):
        dut.data_in[r].value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_reset(dut):
    """After reset, all outputs should be zero."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    for r in range(ROWS):
        assert dut.data_out[r].value == 0, f"data_out[{r}] not zero after reset"
    assert dut.acc_out_bottom.value == 0, "acc_out_bottom not zero after reset"


@cocotb.test()
async def test_weight_load(dut):
    """Loading a weight via acc_in_top should cascade down the column."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Load weight = 5 into the top PE
    dut.acc_in_top.value = 5
    dut.weight_ld.value = 1
    await RisingEdge(dut.clk)
    dut.weight_ld.value = 0

    # MAC: PE0 weight=5 * data_in[0]=3 + bias 0 = 15
    dut.data_in[0].value = 3
    dut.acc_in_top.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    assert dut.acc_out_bottom.value.to_signed() == 15, (
        f"Expected acc_out_bottom=15, got {dut.acc_out_bottom.value.to_signed()}"
    )


@cocotb.test()
async def test_mac_accumulate(dut):
    """Verify multiply-accumulate with a non-zero bias input."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Load weight = 4
    dut.acc_in_top.value = 4
    dut.weight_ld.value = 1
    await RisingEdge(dut.clk)
    dut.weight_ld.value = 0

    # MAC: 4 * 7 + 10 = 38 (acc_in_top=10 is the bias)
    dut.data_in[0].value = 7
    dut.acc_in_top.value = 10
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    assert dut.acc_out_bottom.value.to_signed() == 38, (
        f"Expected acc_out_bottom=38, got {dut.acc_out_bottom.value.to_signed()}"
    )


@cocotb.test()
async def test_data_passthrough(dut):
    """data_in should be forwarded to data_out (no weight loaded)."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    for r in range(ROWS):
        dut.data_in[r].value = 42 + r
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    for r in range(ROWS):
        assert dut.data_out[r].value.to_signed() == 42 + r, (
            f"Expected data_out[{r}]={42 + r}, got {dut.data_out[r].value.to_signed()}"
        )


@cocotb.test()
async def test_weight_load_sequence(dut):
    """Sequential weight loading over multiple cycles cascades down the column."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Cycle 1: load w1
    dut.acc_in_top.value = 11
    dut.weight_ld.value = 1
    await RisingEdge(dut.clk)

    # Cycle 2: load w0
    dut.acc_in_top.value = 22
    await RisingEdge(dut.clk)
    dut.weight_ld.value = 0

    # After cycle 2: PE0 latched w0 on cycle 2, PE1 latched w1 on cycle 2.
    # acc_out is registered, so after cycle 3: acc_out_bottom = w0 (from PE0)
    await RisingEdge(dut.clk)

    assert dut.acc_out_bottom.value.to_signed() == 22, (
        f"Expected acc_out_bottom=22, got {dut.acc_out_bottom.value.to_signed()}"
    )

    # MAC: PE0: 22 * 2 + 5 = 49; PE1: 11 * 0 + 49 = 49
    dut.data_in[0].value = 2
    dut.data_in[1].value = 0
    dut.acc_in_top.value = 5
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    assert dut.acc_out_bottom.value.to_signed() == 49, (
        f"Expected acc_out_bottom=49, got {dut.acc_out_bottom.value.to_signed()}"
    )


@cocotb.test()
async def test_dot_product(dut):
    """Compute a dot product by loading weights and streaming data."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    rng = np.random.default_rng(42)
    weights = rng.integers(1, 6, size=ROWS, dtype=np.int8)
    data = rng.integers(1, 6, size=ROWS, dtype=np.int8)

    # Load weights: feed reversed so PE[r] ends up with weights[r]
    dut.weight_ld.value = 1
    for w in reversed(weights):
        dut.acc_in_top.value = int(w)
        await RisingEdge(dut.clk)
    dut.weight_ld.value = 0

    # Stream data and bias through the column
    for r in range(ROWS):
        dut.data_in[r].value = int(data[r])
    dut.acc_in_top.value = 0  # bias

    # Wait for result to propagate through the column
    for _ in range(ROWS):
        await RisingEdge(dut.clk)

    expected = int(np.dot(weights.astype(np.int32), data.astype(np.int32)))
    got = dut.acc_out_bottom.value.to_signed()
    assert got == expected, f"Dot product: expected {expected}, got {got}"


def run_tests():
    """Build and run all cocotb tests using the cocotb runner API."""
    from cocotb_tools.runner import get_runner

    proj_dir = Path(__file__).resolve().parent.parent
    sources = sorted(proj_dir.glob("src/*.sv"))

    build_dir = Path("sim_build_col")

    runner = get_runner("icarus")
    runner.build(
        sources=sources,
        hdl_toplevel="pe_col",
        build_args=["-g2012", f"-Ppe_col.ROWS={ROWS}"],
        waves=True,
        build_dir=build_dir,
        always=True,
    )
    runner.test(
        hdl_toplevel="pe_col",
        test_module="test_col",
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
