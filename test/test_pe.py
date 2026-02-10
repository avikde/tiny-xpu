import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


async def reset_dut(dut):
    """Apply active-low reset for a few cycles."""
    dut.rst_n.value = 0
    dut.en.value = 0
    dut.weight_ld.value = 0
    dut.data_in.value = 0
    dut.weight_in.value = 0
    dut.acc_in.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_reset(dut):
    """After reset, all outputs should be zero."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    assert dut.data_out.value == 0, "data_out not zero after reset"
    assert dut.acc_out.value == 0, "acc_out not zero after reset"


@cocotb.test()
async def test_weight_load(dut):
    """Loading a weight should latch the value."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    dut.weight_in.value = 5
    dut.weight_ld.value = 1
    await RisingEdge(dut.clk)
    dut.weight_ld.value = 0
    await RisingEdge(dut.clk)

    # Enable a MAC cycle: 5 * 3 + 0 = 15
    dut.data_in.value = 3
    dut.acc_in.value = 0
    dut.en.value = 1

    # NOTE: this edge does the computation, but cocotb timing nuance requires another cycle for the output to be available.
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)  # output registered

    assert dut.acc_out.value.to_signed() == 15, (
        f"Expected acc_out=15, got {dut.acc_out.value.to_signed()}"
    )


@cocotb.test()
async def test_mac_accumulate(dut):
    """Verify multiply-accumulate with a non-zero partial sum input."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Load weight = 4
    dut.weight_in.value = 4
    dut.weight_ld.value = 1
    await RisingEdge(dut.clk)
    dut.weight_ld.value = 0

    # MAC: 4 * 7 + 10 = 38
    dut.data_in.value = 7
    dut.acc_in.value = 10
    dut.en.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    assert dut.acc_out.value.to_signed() == 38, (
        f"Expected acc_out=38, got {dut.acc_out.value.to_signed()}"
    )


@cocotb.test()
async def test_data_passthrough(dut):
    """data_in should be forwarded to data_out when enabled."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    dut.data_in.value = 42
    dut.en.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    assert dut.data_out.value.to_signed() == 42, (
        f"Expected data_out=42, got {dut.data_out.value.to_signed()}"
    )


@cocotb.test()
async def test_enable_gating(dut):
    """When en=0, outputs should hold their previous values."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Do one enabled cycle
    dut.weight_in.value = 2
    dut.weight_ld.value = 1
    await RisingEdge(dut.clk)
    dut.weight_ld.value = 0

    dut.data_in.value = 3
    dut.acc_in.value = 0
    dut.en.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    prev_acc = dut.acc_out.value.to_signed()

    # Disable and change inputs — output should not change
    dut.en.value = 0
    dut.data_in.value = 99
    dut.acc_in.value = 99
    await ClockCycles(dut.clk, 3)

    assert dut.acc_out.value.to_signed() == prev_acc, (
        "acc_out changed while en=0"
    )


def run_tests():
    """Build and run all cocotb tests using the cocotb runner API."""
    from cocotb_tools.runner import get_runner

    proj_dir = Path(__file__).resolve().parent.parent
    sources = sorted(proj_dir.glob("src/*.sv"))

    runner = get_runner("icarus")
    runner.build(
        sources=sources,
        hdl_toplevel="pe",
        build_args=["-g2012"],
        waves=True,
    )
    runner.test(
        hdl_toplevel="pe",
        test_module="test_pe",
        waves=True,
    )

    # cocotb exits 0 even when tests fail; check results XML explicitly
    results_file = Path("sim_build/results.xml")
    if results_file.exists():
        tree = ET.parse(results_file)
        failures = tree.findall(".//failure") + tree.findall(".//error")
        if failures:
            sys.exit(1)


if __name__ == "__main__":
    run_tests()
