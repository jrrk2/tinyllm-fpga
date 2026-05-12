// tb_rope.cpp — Verilator testbench for rope.sv.

#include "Vrope.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <cmath>

static Vrope* dut = nullptr;
static uint64_t cycle = 0;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vrope;

    std::ifstream f("test_rope.bin", std::ios::binary);
    if (!f) { fprintf(stderr, "FAIL: missing test_rope.bin\n"); return 1; }

    uint32_t HEAD_DIM, pos;
    double   base;
    f.read((char*)&HEAD_DIM, sizeof(HEAD_DIM));
    f.read((char*)&pos,      sizeof(pos));
    f.read((char*)&base,     sizeof(base));
    if (HEAD_DIM != 64) { fprintf(stderr, "FAIL: tb expects HEAD_DIM=64, got %u\n", HEAD_DIM); return 1; }

    std::vector<int16_t> x(HEAD_DIM), y_ref(HEAD_DIM);
    f.read((char*)x.data(),     HEAD_DIM * 2);
    f.read((char*)y_ref.data(), HEAD_DIM * 2);

    // Reset
    dut->rst = 1;
    dut->start = 0;
    dut->pos = 0;
    dut->in_x = 0;
    dut->in_valid = 0;
    for (int i = 0; i < 4; i++) tick();
    dut->rst = 0;
    tick();

    // Pulse start with pos
    dut->pos   = pos;
    dut->start = 1;
    tick();
    dut->start = 0;

    // Drive HEAD_DIM cycles of in_x
    for (uint32_t k = 0; k < HEAD_DIM; k++) {
        dut->in_x     = (uint16_t)x[k];
        dut->in_valid = 1;
        tick();
    }
    dut->in_valid = 0;
    dut->in_x     = 0;

    // Collect HEAD_DIM outputs
    std::vector<int16_t> y_dut;
    y_dut.reserve(HEAD_DIM);
    // Timeout increased from HEAD_DIM+64 to HEAD_DIM*20+256 to accommodate the
    // CORDIC-based sin/cos computation (32 pairs × ~20 cycles each = ~640 cycles)
    // that replaced the single-cycle LUT lookup.
    uint64_t timeout = cycle + HEAD_DIM * 20 + 256;
    while (y_dut.size() < HEAD_DIM && cycle < timeout) {
        tick();
        if (dut->out_valid) y_dut.push_back((int16_t)dut->out_y);
    }
    if (y_dut.size() != HEAD_DIM) {
        fprintf(stderr, "FAIL: only got %zu of %u outputs\n", y_dut.size(), HEAD_DIM);
        delete dut;
        return 1;
    }

    int fails = 0, worst_diff = 0;
    for (uint32_t i = 0; i < HEAD_DIM; i++) {
        int diff = (int)y_dut[i] - (int)y_ref[i];
        if (std::abs(diff) > std::abs(worst_diff)) worst_diff = diff;
        if (std::abs(diff) > 1) {
            printf("  lane %3u: dut=%6d  ref=%6d  diff=%+d\n",
                   i, y_dut[i], y_ref[i], diff);
            fails++;
        }
    }
    printf("\n");
    if (fails) {
        printf("FAIL: %d/%u lanes mismatch (>1 LSB), worst diff %+d\n",
               fails, HEAD_DIM, worst_diff);
        delete dut;
        return 1;
    }
    printf("PASS: all %u lanes within +-1 LSB, worst diff %+d\n", HEAD_DIM, worst_diff);
    delete dut;
    return 0;
}
