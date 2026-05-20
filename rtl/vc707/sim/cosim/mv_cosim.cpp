// mv_cosim.cpp — C ABI shim around the Verilator-compiled
// matvec_bfp_engine, so Python can drive it cycle-by-cycle via ctypes
// and compare against a software reference (e.g. host/sim_rtl_fp_hw.py's
// matvec_hw_golden).  Wide signals (w_mant=256-bit, w_exp=128-bit,
// out_mant/out_exp same) are exchanged as uint32_t arrays.

#include "Vmatvec_bfp_engine.h"
#include "verilated.h"
#include <cstdint>
#include <cstring>

namespace {
Vmatvec_bfp_engine* dut = nullptr;
uint64_t            cycle = 0;
}

extern "C" {

void mv_create(void) {
    if (dut) return;
    dut = new Vmatvec_bfp_engine;
    cycle = 0;
    // Power-on: clk=0, rst=1, hold for several ticks.
    dut->clk = 0; dut->rst = 1;
    dut->start_matvec = 0; dut->in_x_mant = 0; dut->in_x_exp = 0;
    dut->in_valid = 0; dut->last_elem = 0;
    std::memset(&dut->w_mant, 0, 32);
    std::memset(&dut->w_exp,  0, 16);
    dut->eval();
}

void mv_destroy(void) {
    delete dut;
    dut = nullptr;
}

void mv_reset(void) {
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

// One clock tick.  Inputs are applied to the DUT's pre-clock side; the
// outputs returned reflect the state AFTER the rising edge has settled.
// w_mant_words: 8 × uint32 (lane 0 in word[0:0], lane 15 in word[7][31:16]).
// w_exp_words:  4 × uint32 (lane 0 in word[0:7], lane 15 in word[3][31:24]).
// out_mant_words / out_exp_words: same layout, written back.
void mv_tick(
    uint8_t  rst,
    uint8_t  start_matvec,
    int16_t  in_x_mant,
    int8_t   in_x_exp,
    uint8_t  in_valid,
    uint8_t  last_elem,
    const uint32_t* w_mant_words,
    const uint32_t* w_exp_words,
    uint8_t* out_valid,
    uint32_t* out_mant_words,
    uint32_t* out_exp_words)
{
    if (!dut) return;
    dut->rst          = rst;
    dut->start_matvec = start_matvec;
    dut->in_x_mant    = static_cast<uint16_t>(in_x_mant);
    dut->in_x_exp     = static_cast<uint8_t> (in_x_exp);
    dut->in_valid     = in_valid;
    dut->last_elem    = last_elem;
    std::memcpy(&dut->w_mant, w_mant_words, 32);
    std::memcpy(&dut->w_exp,  w_exp_words,  16);
    // One rising edge.
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
    *out_valid = dut->out_valid;
    std::memcpy(out_mant_words, &dut->out_mant, 32);
    std::memcpy(out_exp_words,  &dut->out_exp,  16);
}

uint64_t mv_cycle(void) { return cycle; }

}  // extern "C"

// Verilator needs a stub for sc_time_stamp() (unused outside SC).
double sc_time_stamp() { return 0; }
