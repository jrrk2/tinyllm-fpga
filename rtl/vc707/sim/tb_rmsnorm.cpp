// tb_rmsnorm.cpp — Verilator testbench for rmsnorm.sv.

#include "Vrmsnorm.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <cmath>

static Vrmsnorm* dut = nullptr;
static uint64_t cycle = 0;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vrmsnorm;

    std::ifstream f("test_rmsnorm.bin", std::ios::binary);
    if (!f) { fprintf(stderr, "FAIL: missing test_rmsnorm.bin\n"); return 1; }

    uint32_t D;
    double   eps;
    f.read((char*)&D,   sizeof(D));
    f.read((char*)&eps, sizeof(eps));
    if (D != 64) { fprintf(stderr, "FAIL: tb expects D=64, got %u\n", D); return 1; }

    std::vector<int16_t> x(D), g(D), y_ref(D);
    f.read((char*)x.data(),     D * 2);
    f.read((char*)g.data(),     D * 2);
    f.read((char*)y_ref.data(), D * 2);

    // Reset
    dut->rst = 1;
    dut->start = 0;
    dut->in_x = 0;
    dut->in_gamma = 0;
    dut->in_valid = 0;
    for (int i = 0; i < 4; i++) tick();
    dut->rst = 0;
    tick();

    // Pulse start
    dut->start = 1;
    tick();
    dut->start = 0;

    // Drive D cycles of in_x / in_gamma with in_valid
    for (uint32_t k = 0; k < D; k++) {
        dut->in_x     = (uint16_t)x[k];
        dut->in_gamma = (uint16_t)g[k];
        dut->in_valid = 1;
        tick();
    }
    dut->in_valid = 0;
    dut->in_x     = 0;
    dut->in_gamma = 0;

    // Wait for out_valid pulses; collect D outputs
    std::vector<int16_t> y_dut;
    y_dut.reserve(D);
    uint64_t timeout = cycle + D + 64;
    while (y_dut.size() < D && cycle < timeout) {
        tick();
        if (dut->out_valid) y_dut.push_back((int16_t)dut->out_y);
    }
    if (y_dut.size() != D) {
        fprintf(stderr, "FAIL: only got %zu of %u outputs (timeout at cycle %lu)\n",
                y_dut.size(), D, (unsigned long)cycle);
        delete dut;
        return 1;
    }

    int fails = 0;
    int worst_diff = 0;
    for (uint32_t i = 0; i < D; i++) {
        int diff = (int)y_dut[i] - (int)y_ref[i];
        if (std::abs(diff) > std::abs(worst_diff)) worst_diff = diff;
        // Tolerate ±1 LSB for rounding differences between numpy and SV.
        if (std::abs(diff) > 1) {
            printf("  lane %3u: dut=%6d  ref=%6d  diff=%+d\n",
                   i, y_dut[i], y_ref[i], diff);
            fails++;
        }
    }
    printf("\n");
    if (fails) {
        printf("FAIL: %d/%u lanes mismatch (>1 LSB), worst diff %+d\n",
               fails, D, worst_diff);
        delete dut;
        return 1;
    }
    printf("PASS: all %u lanes within ±1 LSB, worst diff %+d\n", D, worst_diff);
    delete dut;
    return 0;
}
