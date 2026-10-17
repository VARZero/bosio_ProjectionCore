set root [file normalize [file join [file dirname [info script]] ..]]
file mkdir $root/build
cd $root/build
open_project -reset hls
set_top bosio_projector
add_files $root/hls/projector.cpp -cflags "-I$root/hls"
add_files -tb $root/hls/test_projector.cpp -cflags "-I$root/hls"
open_solution -reset solution -flow_target vivado
set_part xc7z020clg400-1
create_clock -period 10 -name default
config_compile -pipeline_loops 0
csim_design
csynth_design
cosim_design -rtl verilog
export_design -format ip_catalog -vendor bosio.org -library hls -version 1.0
exec python $root/scripts/generate_registers.py
exit
