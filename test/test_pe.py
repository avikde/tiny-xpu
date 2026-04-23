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
    """Loading a weight via acc_in (weight_ld=1) should latch the value."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Load weight = 5: weight_ld=1, acc_in carries the weight
    dut.acc_in.value = 5
    dut.weight_ld.value = 1
    await RisingEdge(dut.clk)
    dut.weight_ld.value = 0

    # MAC: 5 * 3 + 0 = 15 (acc_in is bias; 0 here)
    dut.data_in.value = 3
    dut.acc_in.value = 0
    dut.en.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)  # output registered

    assert dut.acc_out.value.to_signed() == 15, (
        f"Expected acc_out=15, got {dut.acc_out.value.to_signed()}"
    )


@cocotb.test()
async def test_mac_accumulate(dut):
    """Verify multiply-accumulate with a non-zero bias input."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Load weight = 4
    dut.acc_in.value = 4
    dut.weight_ld.value = 1
    await RisingEdge(dut.clk)
    dut.weight_ld.value = 0

    # MAC: 4 * 7 + 10 = 38 (acc_in=10 is the bias)
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
    """data_in should be forwarded to data_out when enabled (no weight loaded)."""
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

    # Load weight = 2
    dut.acc_in.value = 2
    dut.weight_ld.value = 1
    await RisingEdge(dut.clk)
    dut.weight_ld.value = 0

    # MAC: 2 * 3 + 0 = 6
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


@cocotb.test()
async def test_weight_load_sequence(dut):
    """Sequential weight loading over two cycles updates weight_r each cycle."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Cycle 1: load w1
    dut.acc_in.value = 11
    dut.weight_ld.value = 1
    await RisingEdge(dut.clk)

    # Cycle 2: load w0
    dut.acc_in.value = 22
    await RisingEdge(dut.clk)
    dut.weight_ld.value = 0

    # acc_out is registered, so w1 appears after cycle 2's edge and w0 after cycle 3's.
    await RisingEdge(dut.clk)

    # After cycle 3's edge, acc_out holds w0 (forwarded from cycle 2)
    assert dut.acc_out.value.to_signed() == 22, (
        f"Expected acc_out=22, got {dut.acc_out.value.to_signed()}"
    )

    # weight_r latched w0 on cycle 2
    assert dut.weight_r.value.to_signed() == 22, (
        f"Expected weight_r=22, got {dut.weight_r.value.to_signed()}"
    )

    # MAC: 22 * 2 + 5 = 49
    dut.data_in.value = 2
    dut.acc_in.value = 5
    dut.en.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    assert dut.acc_out.value.to_signed() == 49, (
        f"Expected acc_out=49, got {dut.acc_out.value.to_signed()}"
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
