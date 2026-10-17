set root [file normalize [file join [file dirname [info script]] ..]]
set_param general.maxThreads 4
create_project -force projector_memory $root/build/vivado -part xc7z020clg400-1
set_property ip_repo_paths $root/build/hls/solution/impl/ip [current_project]
update_ip_catalog
create_bd_design projection_memory
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "0" Master "Disable" Slave "Disable"} [get_bd_cells ps7_0]
source $root/scripts/ps7_config.tcl
set_property -dict [list CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_USE_S_AXI_HP0 {1} CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100}] [get_bd_cells ps7_0]
create_bd_cell -type ip -vlnv bosio.org:hls:bosio_projector:1.0 projector_0
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/ps7_0/M_AXI_GP0" Clk "Auto"} [get_bd_intf_pins projector_0/s_axi_control]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/projector_0/m_axi_gmem" Clk "Auto"} [get_bd_intf_pins ps7_0/S_AXI_HP0]
assign_bd_address
validate_bd_design
save_bd_design
set bd [get_files projection_memory.bd]
generate_target all $bd
set wrappers [make_wrapper -files $bd -top]
add_files -norecurse $wrappers
set_property top projection_memory_wrapper [current_fileset]
update_compile_order -fileset sources_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation failed"}
open_run impl_1
file mkdir $root/results
report_utilization -file $root/results/utilization.rpt
report_utilization -hierarchical -file $root/results/hierarchy.rpt
report_timing_summary -file $root/results/timing.rpt
report_drc -file $root/results/drc.rpt
set setup [get_timing_paths -delay_type max -max_paths 1]
set hold [get_timing_paths -delay_type min -max_paths 1]
if {[llength $setup]==0 || [llength $hold]==0} {error "Missing timing paths"}
if {[get_property SLACK $setup]<0 || [get_property SLACK $hold]<0} {error "Timing failed; do not deploy"}
file mkdir $root/bitstream
file copy -force $root/build/vivado/projector_memory.runs/impl_1/projection_memory_wrapper.bit $root/bitstream/projection_memory.bit
set handoffs [glob -nocomplain $root/build/vivado/projector_memory.gen/sources_1/bd/projection_memory/hw_handoff/*.hwh]
if {[llength $handoffs]!=1} {error "Expected one hardware handoff"}
file copy -force [lindex $handoffs 0] $root/bitstream/projection_memory.hwh
exit
