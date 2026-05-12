// tb_swiglu.cpp — Verilator testbench for swiglu.sv.

#include "Vswiglu.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <fstream>
#include <cmath>

static Vswiglu* dut = nullptr;

static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vswiglu;

    std::ifstream f("test_swiglu.bin", std::ios::binary);
    if (!f) { fprintf(stderr, "FAIL: missing test_swiglu.bin\n"); return 1; }

    uint32_t N;
    f.read((char*)&N, sizeof(N));
    std::vector<int16_t> g(N), u(N), y_ref(N);
    f.read((char*)g.data(),     N*2);
    f.read((char*)u.data(),     N*2);
    f.read((char*)y_ref.data(), N*2);

    dut->rst = 1;
    dut->in_gate = 0;
    dut->in_up = 0;
    dut->in_valid = 0;
    for (int i = 0; i < 4; i++) tick();
    dut->rst = 0;
    tick();

    std::vector<int16_t> y_dut;
    y_dut.reserve(N);
    // 1-cycle latency: drive in[k], on the next tick out_valid pulses with y[k].
    for (uint32_t k = 0; k < N; k++) {
        dut->in_gate  = (uint16_t)g[k];
        dut->in_up    = (uint16_t)u[k];
        dut->in_valid = 1;
        tick();
        if (dut->out_valid) y_dut.push_back((int16_t)dut->out_y);
    }
    // Drain the last beat
    dut->in_valid = 0; dut->in_gate = 0; dut->in_up = 0;
    tick();
    if (dut->out_valid) y_dut.push_back((int16_t)dut->out_y);

    if (y_dut.size() != N) {
        fprintf(stderr, "FAIL: got %zu of %u outputs\n", y_dut.size(), N);
        delete dut;
        return 1;
    }

    int fails = 0, worst_diff = 0;
    for (uint32_t i = 0; i < N; i++) {
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
               fails, N, worst_diff);
        delete dut;
        return 1;
    }
    printf("PASS: all %u lanes within +-1 LSB, worst diff %+d\n", N, worst_diff);
    delete dut;
    return 0;
}
