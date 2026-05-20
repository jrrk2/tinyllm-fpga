// sw_cosim.cpp — C ABI shim around the Verilator-compiled swiglu_bfp.
// All inputs/outputs are scalar (≤16-bit), so no wide-signal marshalling
// is needed.

#include "Vswiglu_bfp.h"
#include "verilated.h"
#include <cstdint>

namespace {
Vswiglu_bfp* dut = nullptr;
uint64_t     cycle = 0;
}

extern "C" {

void sw_create(void) {
    if (dut) return;
    dut = new Vswiglu_bfp;
    cycle = 0;
    dut->clk = 0; dut->rst = 1;
    dut->start = 0; dut->in_gate_mant = 0; dut->in_gate_exp = 0;
    dut->in_up_mant = 0; dut->in_up_exp = 0;
    dut->in_valid = 0; dut->last_elem = 0;
    dut->eval();
}

void sw_destroy(void) { delete dut; dut = nullptr; }

void sw_reset(void) {
    if (!dut) return;
    dut->rst = 1;
    for (int i = 0; i < 4; ++i) {
        dut->clk = 0; dut->eval();
        dut->clk = 1; dut->eval();
        cycle++;
    }
    dut->rst = 0;
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
}

// One clock tick.  Outputs reflect state AFTER the rising edge.
void sw_tick(
    uint8_t  rst,
    uint8_t  start,
    int16_t  in_gate_mant,
    int8_t   in_gate_exp,
    int16_t  in_up_mant,
    int8_t   in_up_exp,
    uint8_t  in_valid,
    uint8_t  last_elem,
    uint8_t* in_ready,
    int16_t* out_y_mant,
    int8_t*  out_y_exp,
    uint8_t* out_valid,
    uint8_t* done)
{
    if (!dut) return;
    dut->rst          = rst;
    dut->start        = start;
    dut->in_gate_mant = static_cast<uint16_t>(in_gate_mant);
    dut->in_gate_exp  = static_cast<uint8_t> (in_gate_exp);
    dut->in_up_mant   = static_cast<uint16_t>(in_up_mant);
    dut->in_up_exp    = static_cast<uint8_t> (in_up_exp);
    dut->in_valid     = in_valid;
    dut->last_elem    = last_elem;
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
    *in_ready   = dut->in_ready;
    *out_y_mant = static_cast<int16_t>(dut->out_y_mant);
    *out_y_exp  = static_cast<int8_t> (dut->out_y_exp);
    *out_valid  = dut->out_valid;
    *done       = dut->done;
}

uint64_t sw_cycle(void) { return cycle; }

}  // extern "C"

double sc_time_stamp() { return 0; }
