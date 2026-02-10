cmake_minimum_required(VERSION 3.16)
project(tiny-xpu LANGUAGES NONE)

# ---------- tool discovery ----------
find_program(IVERILOG iverilog REQUIRED)
find_program(VVP vvp REQUIRED)
find_program(YOSYS yosys)

# ---------- collect all .sv sources ----------
file(GLOB_RECURSE SV_SOURCES "${CMAKE_SOURCE_DIR}/src/*.sv")

# ---------- lint / compile check ----------
set(COMPILED_VVP "${CMAKE_BINARY_DIR}/tiny_xpu.vvp")

add_custom_command(
    OUTPUT  ${COMPILED_VVP}
    COMMAND ${IVERILOG} -g2012 -Wall -o ${COMPILED_VVP} ${SV_SOURCES}
    DEPENDS ${SV_SOURCES}
    COMMENT "Compiling SystemVerilog sources with Icarus Verilog"
)

add_custom_target(compile ALL DEPENDS ${COMPILED_VVP})

# ---------- optional: Yosys synthesis ----------
if(YOSYS)
    set(SYNTH_JSON "${CMAKE_BINARY_DIR}/synth.json")

    # Build a read_verilog command for each source file
    set(YOSYS_READ_CMDS "")
    foreach(src ${SV_SOURCES})
        string(APPEND YOSYS_READ_CMDS "read_verilog -sv ${src}; ")
    endforeach()

    add_custom_command(
        OUTPUT  ${SYNTH_JSON}
        COMMAND ${YOSYS} -p
                "${YOSYS_READ_CMDS} synth -top pe; write_json ${SYNTH_JSON}"
        DEPENDS ${SV_SOURCES}
        COMMENT "Synthesising with Yosys"
    )

    add_custom_target(synth DEPENDS ${SYNTH_JSON})
endif()

# ---------- cocotb tests (run via: ctest from build/) ----------
# Each test/test_*.py uses cocotb_tools.runner to compile and simulate.
find_program(PYTHON3 python3 REQUIRED)

enable_testing()

file(GLOB TEST_FILES "${CMAKE_SOURCE_DIR}/test/test_*.py")
foreach(test_file ${TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_test(
        NAME    ${test_name}
        COMMAND ${PYTHON3} ${test_file}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/test"
    )
endforeach()

