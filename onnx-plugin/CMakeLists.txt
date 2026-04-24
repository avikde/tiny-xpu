cmake_minimum_required(VERSION 3.18)
project(sample_onnx_ep VERSION 1.0.0 LANGUAGES CXX)

option(SIM "Link Verilator-compiled SV into the ONNX EP" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Find ONNX Runtime
# ============================================================================

set(ONNXRUNTIME_INCLUDE_DIR "/opt/onnxruntime/include")
set(ONNXRUNTIME_LIB_DIR "/opt/onnxruntime/lib")

message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib dir: ${ONNXRUNTIME_LIB_DIR}")

# ============================================================================
# Build the Plugin EP Shared Library
# ============================================================================

add_library(tinyxpu_ep SHARED
    src/tinyxpu_ep.cpp
    src/tinyxpu_driver.cpp
    src/tinyxpu_perf.cpp
)

target_include_directories(tinyxpu_ep PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# Set visibility to hidden by default, only export what we need
if(NOT MSVC)
  target_compile_options(tinyxpu_ep PRIVATE
        -fvisibility=hidden
        -Wall
        -Wextra
    )
endif()

# Platform-specific settings
if(WIN32)
  set_target_properties(tinyxpu_ep PROPERTIES
        PREFIX ""
        OUTPUT_NAME "tinyxpu_ep"
        SUFFIX ".dll"
    )
elseif(APPLE)
  set_target_properties(tinyxpu_ep PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "tinyxpu_ep"
        SUFFIX ".dylib"
    )
else()
  set_target_properties(tinyxpu_ep PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "tinyxpu_ep"
        SUFFIX ".so"
    )
endif()

# ============================================================================
# Optional: Verilator simulation backend (-DSIM=ON)
# ============================================================================

if(SIM)
  find_package(verilator HINTS $ENV{VERILATOR_ROOT} /usr/share/verilator)
  if(NOT verilator_FOUND)
    message(FATAL_ERROR "SIM=ON but Verilator not found")
  endif()

  set(SIM_ROWS 16 CACHE STRING "Systolic array row count")
  set(SIM_COLS 16 CACHE STRING "Systolic array column count")

  verilate(tinyxpu_ep
        SOURCES
            ${CMAKE_SOURCE_DIR}/src/pe.sv
            ${CMAKE_SOURCE_DIR}/src/pe_col.sv
            ${CMAKE_SOURCE_DIR}/src/requant.sv
            ${CMAKE_SOURCE_DIR}/src/array.sv
        TOP_MODULE  array
        PREFIX      Varray
        VERILATOR_ARGS -GROWS=${SIM_ROWS} -GCOLS=${SIM_COLS}
    )

  target_compile_definitions(tinyxpu_ep PRIVATE
        TINYXPU_USE_VERILATOR
        TINYXPU_ARRAY_ROWS=${SIM_ROWS}
        TINYXPU_ARRAY_COLS=${SIM_COLS}
    )

  # Mark Verilator's own headers as SYSTEM so the compiler doesn't emit
  # warnings from generated/runtime code.
  target_include_directories(tinyxpu_ep SYSTEM PRIVATE
        ${VERILATOR_ROOT}/include
        ${VERILATOR_ROOT}/include/vltstd
    )

  message(STATUS "SIM=ON: Verilator ${verilator_VERSION}")
  message(STATUS "Array size ${SIM_ROWS}x${SIM_COLS}")
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS tinyxpu_ep
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
