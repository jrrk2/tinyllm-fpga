// rp_cosim.cpp — C ABI shim around the Verilator-compiled rope_bfp.
// All scalar signals; no wide-signal marshalling needed.

#include "Vrope_bfp.h"
#include "verilated.h"
#include <cstdint>

namespace { Vrope_bfp* dut = nullptr; uint64_t cycle = 0; }

extern "C" {

void rp_create(void) {
    if (dut) return;
    dut = new Vrope_bfp;
    cycle = 0;
    dut->clk = 0; dut->rst = 1;
    dut->start = 0; dut->in_x_mant = 0; dut->in_x_exp = 0;
    dut->in_valid = 0; dut->pos = 0;
    dut->eval();
}
void rp_destroy(void) { delete dut; dut = nullptr; }

void rp_reset(void) {
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

void rp_tick(
    uint8_t  rst,
    uint8_t  start,
    int16_t  in_x_mant,
    int8_t   in_x_exp,
    uint8_t  in_valid,
    uint16_t pos,
    int16_t* out_y_mant,
    int8_t*  out_y_exp,
    uint8_t* out_valid,
    uint8_t* done)
{
    if (!dut) return;
    dut->rst       = rst;
    dut->start     = start;
    dut->in_x_mant = static_cast<uint16_t>(in_x_mant);
    dut->in_x_exp  = static_cast<uint8_t> (in_x_exp);
    dut->in_valid  = in_valid;
    dut->pos       = pos;
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
    *out_y_mant = static_cast<int16_t>(dut->out_y_mant);
    *out_y_exp  = static_cast<int8_t> (dut->out_y_exp);
    *out_valid  = dut->out_valid;
    *done       = dut->done;
}

uint64_t rp_cycle(void) { return cycle; }

}  // extern "C"

double sc_time_stamp() { return 0; }
