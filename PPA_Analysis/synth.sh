#!/bin/bash
yosys -p "
    read_verilog results/sv2v_output.v;
    chparam -set ROWS $ROWS \
            -set COLS $COLS \
            -set DATA_WIDTH $DATA_WIDTH \
            -set ACC_WIDTH $ACC_WIDTH \
            $TOP_MODULE
    synth -top $TOP_MODULE;
    dfflibmap -liberty $SKYWATER_LIB;
    abc -liberty $SKYWATER_LIB;
    stat -liberty $SKYWATER_LIB;
    rename -top $TOP_MODULE
    write_verilog gate_level.v
" > SYNTH.log
