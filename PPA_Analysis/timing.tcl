read_liberty $::env(SKYWATER_LIB)

read_verilog gate_level.v

link_design $::env(TOP_MODULE)

create_clock -period $::env(CLK_PERIOD) [get_ports clk]

report_checks -path_delay max -group_count 5

puts "Timing Analysis Completed."

exit
