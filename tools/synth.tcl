set root [file normalize [file join [file dirname [info script]] ..]]
file mkdir [file join $root build synth]
cd [file join $root rtl generated]
create_project -in_memory -part xc7a100tcsg324-1
set_property include_dirs [list [pwd]] [current_fileset]
read_verilog -sv [glob [file join $root rtl *.sv]]
synth_design -top wb_cnn -part xc7a100tcsg324-1
create_clock -period 10 [get_ports i_clk]
report_utilization -file [file join $root build synth utilization.rpt]
report_timing_summary -file [file join $root build synth timing.rpt]
write_checkpoint -force [file join $root build synth cnn.dcp]
