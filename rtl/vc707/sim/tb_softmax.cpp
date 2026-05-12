// tb_softmax.cpp — Verilator testbench for softmax_q15.sv.

#include "Vsoftmax_q15.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <fstream>
#include <cmath>

static Vsoftmax_q15* dut = nullptr;
static uint64_t cycle = 0;

static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); cycle++; }

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vsoftmax_q15;

    std::ifstream f("test_softmax.bin", std::ios::binary);
    if (!f) { fprintf(stderr, "FAIL: missing test_softmax.bin\n"); return 1; }

    uint32_t N;
    f.read((char*)&N, sizeof(N));
    std::vector<int16_t> x(N), y_ref(N);
    f.read((char*)x.data(),     N*2);
    f.read((char*)y_ref.data(), N*2);

    dut->rst = 1;
    dut->start = 0;
    dut->in_x = 0;
    dut->in_valid = 0;
    for (int i = 0; i < 4; i++) tick();
    dut->rst = 0;
    tick();

    dut->start = 1;
    tick();
    dut->start = 0;

    for (uint32_t k = 0; k < N; k++) {
        dut->in_x     = (uint16_t)x[k];
        dut->in_valid = 1;
        tick();
    }
    dut->in_valid = 0;
    dut->in_x     = 0;

    std::vector<int16_t> y_dut;
    y_dut.reserve(N);
    uint64_t timeout = cycle + 3*N + 32;  // allow S_EXP(N) + S_RECIP(6) + S_OUTPUT(N)
    while (y_dut.size() < N && cycle < timeout) {
        tick();
        if (dut->out_valid) y_dut.push_back((int16_t)dut->out_y);
    }
    if (y_dut.size() != N) {
        fprintf(stderr, "FAIL: got %zu of %u outputs\n", y_dut.size(), N);
        delete dut;
        return 1;
    }

    int fails = 0, worst_diff = 0;
    int64_t sum_dut = 0;
    for (uint32_t i = 0; i < N; i++) {
        sum_dut += y_dut[i];
        int diff = (int)y_dut[i] - (int)y_ref[i];
        if (std::abs(diff) > std::abs(worst_diff)) worst_diff = diff;
        // Tolerate ±10 LSB: LUT quantisation + NR reciprocal error.
        if (std::abs(diff) > 10) {
            printf("  lane %3u: dut=%6d  ref=%6d  diff=%+d\n",
                   i, y_dut[i], y_ref[i], diff);
            fails++;
        }
    }
    printf("\nDUT sum = %ld  (Q1.15 unit = 32768)\n", (long)sum_dut);
    if (fails) {
        printf("FAIL: %d/%u lanes mismatch (>10 LSB), worst diff %+d\n",
               fails, N, worst_diff);
        delete dut;
        return 1;
    }
    printf("PASS: all %u lanes within +-10 LSB, worst diff %+d\n", N, worst_diff);
    delete dut;
    return 0;
}
