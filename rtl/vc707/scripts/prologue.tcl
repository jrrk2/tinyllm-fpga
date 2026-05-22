# Project name is per-build (target + date stamp) so different tops never share
# runs (no cross-top "synth_1 needs reset") and old results survive — you never
# need to wipe.  Set by the Makefile via VPROJECT; falls back to the legacy fixed
# name when unset (e.g. plain `make program`, which only reads work/).
set project [expr {[info exists ::env(VPROJECT)] ? $::env(VPROJECT) : "microgpt_eth"}]

# Open existing project if present (preserves runs/cache/etc.), else
# create a fresh one.  -force was deleting microgpt_eth.runs/ on every
# `make program` because prologue.tcl runs before program.tcl too.
# Use `make clean` to start over.
if {[file exists "$project.xpr"]} {
  open_project $project.xpr
} else {
  create_project $project . -part $::env(XILINX_PART)
  set_property board_part $::env(XILINX_BOARD) [current_project]
}

set_param general.maxThreads 8

set_msg_config -id {[Synth 8-5858]} -new_severity "info"
set_msg_config -id {[Synth 8-4480]} -limit 1000
