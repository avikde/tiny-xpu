# Yosys synthesis target for the systolic array.
#
# Yosys does not support unpacked array module ports (used in array.sv and
# pe_col.sv).  This file generates a flat module that directly instantiates
# every PE with individual scalar wires — no unpacked arrays anywhere.
#
# Exposes a `synth` target that runs:
#   read_verilog → hierarchy → flatten → proc → techmap →
#   dfflibmap → abc -liberty (stime -p timing report) → opt_clean →
#   write_json → stat
#
# Requires $SKYWATER_LIB env var pointing to the SkyWater130 Liberty .lib file.
# Timing estimates, cell statistics, and gate-level netlist are written to
# <project>/synth_outputs/.

if(NOT YOSYS)
    return()
endif()

# --- SkyWater130 Liberty library (env var only, no caching) ---
if(NOT DEFINED ENV{SKYWATER_LIB} OR "$ENV{SKYWATER_LIB}" STREQUAL "")
  message(FATAL_ERROR
    "Set the SKYWATER_LIB env var to the path of sky130_fd_sc_hd__tt_025C_1v80.lib.\n"
    "See README.md for download instructions.")
endif()
if(NOT EXISTS "$ENV{SKYWATER_LIB}")
  message(FATAL_ERROR "SKYWATER_LIB=$ENV{SKYWATER_LIB} does not exist")
endif()

# --- Clock period for timing-driven mapping ---
set(CLOCK_PERIOD_NS 10 CACHE STRING "Clock period in ns for ABC timing-driven mapping")

# --- Output directory for synthesis results ---
set(SYNTH_OUT "${CMAKE_SOURCE_DIR}/synth_outputs")
file(MAKE_DIRECTORY ${SYNTH_OUT})

# --- ABC timing script (generated at configure time) ---
set(ABC_SCRIPT "${CMAKE_BINARY_DIR}/abc_timing.script")
file(WRITE ${ABC_SCRIPT}
    "strash; &get -n; &fraig -x; &put; scorr; dc2; dretime; strash; &get -n; &dch -f; &nf {D}; &put; stime -p\n")

set(SYNTH_JSON "${SYNTH_OUT}/synth.json")

# Cache variables for array dimensions (shared with onnx-plugin).
set(SIM_ROWS 16 CACHE STRING "Systolic array row count")
set(SIM_COLS 16 CACHE STRING "Systolic array column count")

set(SYNTH_SRC "${CMAKE_BINARY_DIR}/synth_array.sv")
set(DW 8)
set(AW 32)
math(EXPR MAX_R "${SIM_ROWS} - 1")
math(EXPR MAX_C "${SIM_COLS} - 1")

file(WRITE ${SYNTH_SRC}
    "// Auto-generated flat array for Yosys synthesis\n"
    "// Array: ${SIM_ROWS}x${SIM_COLS}, DATA_WIDTH=${DW}, ACC_WIDTH=${AW}\n"
    "`timescale 1ns / 1ps\n\n"
    "module synth_array (\n"
    "    input  logic clk,\n"
    "    input  logic rst_n")

foreach(r RANGE 0 ${MAX_R})
    file(APPEND ${SYNTH_SRC}
        ",\n    input  logic signed [${DW}-1:0] data_in_left_${r}")
endforeach()
foreach(r RANGE 0 ${MAX_R})
    file(APPEND ${SYNTH_SRC}
        ",\n    output logic signed [${DW}-1:0] data_out_right_${r}")
endforeach()
foreach(c RANGE 0 ${MAX_C})
    file(APPEND ${SYNTH_SRC}
        ",\n    input  logic weight_ld_${c}")
endforeach()
foreach(c RANGE 0 ${MAX_C})
    file(APPEND ${SYNTH_SRC}
        ",\n    input  logic signed [${AW}-1:0] acc_in_top_${c}")
endforeach()
foreach(c RANGE 0 ${MAX_C})
    file(APPEND ${SYNTH_SRC}
        ",\n    output logic signed [${AW}-1:0] acc_out_bottom_${c}")
endforeach()

file(APPEND ${SYNTH_SRC} "\n);\n\n")

# Data wires across rows (n + 1 columns: 0..COLS)
foreach(c RANGE 0 ${SIM_COLS})
    foreach(r RANGE 0 ${MAX_R})
        file(APPEND ${SYNTH_SRC}
            "    logic signed [${DW}-1:0] data_wire_${c}_${r};\n")
    endforeach()
endforeach()
file(APPEND ${SYNTH_SRC} "\n")

# Acc wires down columns (n + 1 rows per column: 0..ROWS)
foreach(c RANGE 0 ${MAX_C})
    foreach(r RANGE 0 ${SIM_ROWS})
        file(APPEND ${SYNTH_SRC}
            "    logic signed [${AW}-1:0] acc_wire_${c}_${r};\n")
    endforeach()
endforeach()
file(APPEND ${SYNTH_SRC} "\n")

# Left-edge data inputs
foreach(r RANGE 0 ${MAX_R})
    file(APPEND ${SYNTH_SRC}
        "    assign data_wire_0_${r} = data_in_left_${r};\n")
endforeach()
file(APPEND ${SYNTH_SRC} "\n")

# Right-edge data outputs
foreach(r RANGE 0 ${MAX_R})
    file(APPEND ${SYNTH_SRC}
        "    assign data_out_right_${r} = data_wire_${SIM_COLS}_${r};\n")
endforeach()
file(APPEND ${SYNTH_SRC} "\n")

# Top-edge acc inputs
foreach(c RANGE 0 ${MAX_C})
    file(APPEND ${SYNTH_SRC}
        "    assign acc_wire_${c}_0 = acc_in_top_${c};\n")
endforeach()
file(APPEND ${SYNTH_SRC} "\n")

# Bottom-edge acc outputs
foreach(c RANGE 0 ${MAX_C})
    file(APPEND ${SYNTH_SRC}
        "    assign acc_out_bottom_${c} = acc_wire_${c}_${SIM_ROWS};\n")
endforeach()
file(APPEND ${SYNTH_SRC} "\n")

# Instantiate all PEs
foreach(c RANGE 0 ${MAX_C})
    math(EXPR NEXT_C "${c} + 1")
    foreach(r RANGE 0 ${MAX_R})
        math(EXPR NEXT_R "${r} + 1")
        file(APPEND ${SYNTH_SRC}
            "    pe #(\n"
            "        .DATA_WIDTH (${DW}),\n"
            "        .ACC_WIDTH  (${AW})\n"
            "    ) u_pe_${c}_${r} (\n"
            "        .clk       (clk),\n"
            "        .rst_n     (rst_n),\n"
            "        .weight_ld (weight_ld_${c}),\n"
            "        .data_in   (data_wire_${c}_${r}),\n"
            "        .data_out  (data_wire_${NEXT_C}_${r}),\n"
            "        .acc_in    (acc_wire_${c}_${r}),\n"
            "        .acc_out   (acc_wire_${c}_${NEXT_R})\n"
            "    );\n")
    endforeach()
endforeach()

file(APPEND ${SYNTH_SRC} "\nendmodule\n")

add_custom_command(
    OUTPUT  ${SYNTH_JSON}
    COMMAND ${YOSYS}
        -p "read_verilog -sv ${CMAKE_SOURCE_DIR}/src/pe.sv ${SYNTH_SRC}"
        -p "hierarchy -check -top synth_array"
        -p "flatten"
        -p "proc"
        -p "opt"
        -p "techmap"
        -p "opt"
        -p "dfflibmap -liberty $ENV{SKYWATER_LIB}"
        -p "tee -o ${SYNTH_OUT}/synth_timing.rpt abc -liberty $ENV{SKYWATER_LIB} -script ${CMAKE_BINARY_DIR}/abc_timing.script"
        -p "opt_clean"
        -p "write_json ${SYNTH_JSON}"
        -p "tee -o ${SYNTH_OUT}/synth_stats.txt stat -width"
    DEPENDS
        ${CMAKE_SOURCE_DIR}/src/pe.sv
        ${SYNTH_SRC}
        ${ABC_SCRIPT}
    COMMENT "Synthesising array (${SIM_ROWS}x${SIM_COLS}) to SkyWater130 cells"
)

add_custom_target(synth DEPENDS ${SYNTH_JSON})
