#!/bin/bash
./sv2v.sh
for DW in 4 8 16 32
do
    for AW in 16 32 64 128
    do
        for N in 4 8 16 32
        do
            export ROWS=$N
            export COLS=$N
            export DATA_WIDTH=$DW
            export ACC_WIDTH=$AW
            ./synth.sh
            echo "Synthesis completed."
            sta timing.tcl > TIMING.log
            echo "STA completed."
            openroad power.tcl > POWER.log
            echo "Power Analysis completed."
            mkdir results/R${N}_C${N}_DW${DW}_AW${AW}
            mv gate_level.v results/R${N}_C${N}_DW${DW}_AW${AW}/
            mv SYNTH.log results/R${N}_C${N}_DW${DW}_AW${AW}/
            mv TIMING.log results/R${N}_C${N}_DW${DW}_AW${AW}/
            mv POWER.log results/R${N}_C${N}_DW${DW}_AW${AW}/
            echo "----------------"
        done
    done
done
