// tb_matvec.cpp — Verilator testbench for matvec_int8_engine.
//
// Reads test_matvec.bin produced by gen_matvec_test.py, drives the engine,
// and checks the per-lane output against the host's reference.

#include "Vmatvec_int8_engine.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>

static Vmatvec_int8_engine* dut = nullptr;
static uint64_t main_time = 0;

static void tick() {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vmatvec_int8_engine;

    // Load test vectors
    std::ifstream f("test_matvec.bin", std::ios::binary);
    if (!f) { fprintf(stderr, "FAIL: missing test_matvec.bin\n"); return 1; }

    uint32_t lanes, in_dim;
    f.read((char*)&lanes,  sizeof(lanes));
    f.read((char*)&in_dim, sizeof(in_dim));
    if (lanes != 16) { fprintf(stderr, "FAIL: tb compiled for LANES=16, got %u\n", lanes); return 1; }

    std::vector<int16_t> in_q15(in_dim);
    f.read((char*)in_q15.data(), in_dim * sizeof(int16_t));

    std::vector<int8_t>  w_packed(in_dim * lanes);  // [k][lane]
    f.read((char*)w_packed.data(), in_dim * lanes);

    std::vector<int16_t> scale_q15(lanes);
    f.read((char*)scale_q15.data(), lanes * sizeof(int16_t));

    std::vector<int16_t> out_ref(lanes);
    f.read((char*)out_ref.data(), lanes * sizeof(int16_t));

    // Reset
    dut->rst = 1;
    dut->in_value = 0; dut->in_valid = 0; dut->in_last = 0;
    dut->scale_valid = 0; dut->acc_clear = 0;
    // w_int8 and scale_q15 are 128-bit and 256-bit packed buses; on Verilator
    // they appear as WData arrays.  Zero them to start.
    memset(&dut->w_int8,    0, sizeof(dut->w_int8));
    memset(&dut->scale_q15, 0, sizeof(dut->scale_q15));
    for (int i = 0; i < 4; i++) tick();

    dut->rst = 0;
    dut->acc_clear = 1;
    tick();
    dut->acc_clear = 0;

    // Drive in_dim cycles of (in_value, w_int8) with in_valid asserted.
    // w_int8 is a 128-bit (16 lanes × 8 bits) packed bus; Verilator exposes
    // it as VlWide<4> on a 16-byte boundary.  We can write it as raw bytes.
    for (uint32_t k = 0; k < in_dim; k++) {
        dut->in_value = (uint16_t)in_q15[k];
        dut->in_valid = 1;
        dut->in_last  = (k == in_dim - 1) ? 1 : 0;
        // Pack 16 lane bytes into the w_int8 bus.
        uint8_t* wb = reinterpret_cast<uint8_t*>(&dut->w_int8);
        for (uint32_t l = 0; l < lanes; l++) {
            wb[l] = (uint8_t) w_packed[k * lanes + l];
        }
        tick();
    }
    dut->in_valid = 0;
    dut->in_last  = 0;
    dut->in_value = 0;

    // The accumulator updates one cycle after in_valid (acc_r <= acc_r + mul_r,
    // where mul_r itself is one cycle behind in_value).  Wait two cycles for
    // the full MAC chain to drain before asserting scale_valid.
    tick();
    tick();

    // Now assert scale_valid, with scales packed.
    {
        uint8_t* sb = reinterpret_cast<uint8_t*>(&dut->scale_q15);
        for (uint32_t l = 0; l < lanes; l++) {
            sb[l*2 + 0] = ((uint16_t)scale_q15[l]) & 0xFF;
            sb[l*2 + 1] = ((uint16_t)scale_q15[l]) >> 8;
        }
    }
    dut->scale_valid = 1;
    tick();
    dut->scale_valid = 0;
    tick();           // out_valid pulses one cycle after scale_valid
    tick();           // settle out_value

    // Read out_value (256 bits = 32 bytes)
    std::vector<int16_t> out_dut(lanes);
    {
        uint8_t* ob = reinterpret_cast<uint8_t*>(&dut->out_value);
        for (uint32_t l = 0; l < lanes; l++) {
            uint16_t v = ob[l*2] | (uint16_t(ob[l*2+1]) << 8);
            out_dut[l] = (int16_t)v;
        }
    }

    int fails = 0;
    printf("lane :    dut       ref     diff\n");
    for (uint32_t l = 0; l < lanes; l++) {
        int diff = (int)out_dut[l] - (int)out_ref[l];
        const char* tag = (diff == 0) ? "ok" : "FAIL";
        printf("%4u : %7d   %7d   %+5d  %s\n",
               l, out_dut[l], out_ref[l], diff, tag);
        if (diff != 0) fails++;
    }

    if (fails) {
        printf("\nFAIL: %d/%u lanes mismatched\n", fails, lanes);
        delete dut;
        return 1;
    }
    printf("\nPASS: all %u lanes match\n", lanes);
    delete dut;
    return 0;
}
