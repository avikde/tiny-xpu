read_lef $::env(SKYWATER_TLEF)
read_lef $::env(SKYWATER_LEF)
read_liberty $::env(SKYWATER_LIB)

read_verilog gate_level.v

link_design $::env(TOP_MODULE)

create_clock -period $::env(CLK_PERIOD) [get_ports clk]

set_power_activity -activity 0.5

report_power

puts "Power Analysis Completed."

exit
