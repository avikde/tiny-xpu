# Set enviroment variable
export PROJECT_HOME=/path/to/tiny-xpu/
export TOP_MODULE=<top-module>
export SKYWATER_LIB=/path/to/sky130_fd_sc_hd__tt_025C_1v80.lib
export SKYWATER_TLEF=/path/to/sky130_fd_sc_hd__nom.tlef
export SKYWATER_LEF=/path/to/sky130_fd_sc_hd.lef
export SV2V_PATH=/path/to/sv2v
export CLK_PERIOD=<clock-period>

# 1. Convert SystemVerilog RTL to Verilog
./sv2v.sh

# 2. Run Synthesis using Yosys to get Area
./synth.sh

# 3. Run STA using OpenSTA to get critical timing paths
./sta.sh

# Remove generated files
./clean.sh

# Run PPA sweep
./sweep.sh
