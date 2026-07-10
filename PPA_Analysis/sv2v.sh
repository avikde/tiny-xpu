#!/bin/bash
$SV2V_PATH/bin/sv2v $PROJECT_HOME/src/pe.sv $PROJECT_HOME/src/pe_col.sv $PROJECT_HOME/src/array.sv $PROJECT_HOME/src/requant.sv --top $TOP_MODULE > results/sv2v_output.v
echo "System Verilog to Verilog conversion complete"
