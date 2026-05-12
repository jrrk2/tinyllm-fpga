// tb_multilayer.cpp — Verilator harness for smollm_multilayer.sv (NL=3).
// Loads multilayer_HIDDEN_IN.hex, drives the cascade, compares the final
// hidden_out against multilayer_expected.txt.

#include "Vsmollm_multilayer.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

static const int D = 128;
static const int NL = 3;

static Vsmollm_multilayer* dut = nullptr;
static uint64_t cycle = 0;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
}

static std::vector<int16_t> load_int16_hex(const char* path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "FAIL: missing %s\n", path); std::exit(1); }
    std::vector<int16_t> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        unsigned v;
        std::istringstream iss(line);
        std::string tok;
        iss >> tok;
        if (std::sscanf(tok.c_str(), "%x", &v) == 1)
            out.push_back((int16_t)v);
    }
    return out;
}

static std::vector<int16_t> load_expected_signed(const char* path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "FAIL: missing %s\n", path); std::exit(1); }
    std::vector<int16_t> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string hex; int dec;
        if (iss >> hex >> dec) out.push_back((int16_t)dec);
    }
    return out;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vsmollm_multilayer;

    auto h_in = load_int16_hex("../generated/multilayer_HIDDEN_IN.hex");
    if ((int)h_in.size() != D) {
        fprintf(stderr, "FAIL: hidden_in size %zu vs D=%d\n", h_in.size(), D);
        return 1;
    }
    auto h_ref = load_expected_signed("../generated/multilayer_expected.txt");
    if ((int)h_ref.size() != D) {
        fprintf(stderr, "FAIL: expected size %zu vs D=%d\n", h_ref.size(), D);
        return 1;
    }

    dut->rst    = 1;
    dut->start  = 0;
    dut->pos    = 3;
    dut->kv_pos = 3;
    {
        uint8_t* hb = reinterpret_cast<uint8_t*>(&dut->hidden_in);
        for (int l = 0; l < D; l++) {
            uint16_t u = (uint16_t)h_in[l];
            hb[l*2 + 0] = u & 0xFF;
            hb[l*2 + 1] = u >> 8;
        }
    }
    for (int i = 0; i < 4; i++) tick();
    dut->rst = 0;
    dut->start = 1;     // hold high for the duration

    // 3 layers × ~11K cycles each + 256 cache init each (parallel during rst)
    // ≈ 35K cycles total.  Be generous with timeout.
    const uint64_t timeout = 100000;
    while (!dut->done && cycle < timeout) tick();
    if (!dut->done) {
        fprintf(stderr, "FAIL: done never asserted (cycle=%lu)\n", (unsigned long)cycle);
        delete dut;
        return 1;
    }
    printf("done=1 at cycle %lu (NL=%d layers)\n\n", (unsigned long)cycle, NL);

    // Compare final hidden_out
    int16_t lanes[D];
    {
        uint8_t* rb = reinterpret_cast<uint8_t*>(&dut->hidden_out);
        for (int l = 0; l < D; l++) {
            uint16_t v = rb[l*2] | (uint16_t(rb[l*2+1]) << 8);
            lanes[l] = (int16_t)v;
        }
    }

    int fails = 0, worst = 0;
    const int TOL = 100;   // 3-layer cumulative noise — wide tolerance
    for (int l = 0; l < D; l++) {
        int diff = (int)lanes[l] - (int)h_ref[l];
        if (std::abs(diff) > std::abs(worst)) worst = diff;
        if (std::abs(diff) > TOL) {
            if (fails < 6)
                printf("  lane %3d: dut=%+6d ref=%+6d diff=%+5d\n",
                       l, lanes[l], h_ref[l], diff);
            fails++;
        }
    }
    if (fails) {
        printf("\nFAIL: %d/%d lanes >±%d LSB, worst %+d\n", fails, D, TOL, worst);
        delete dut;
        return 1;
    }
    printf("PASS: smollm_multilayer (NL=%d, D=%d) hidden_out within ±%d LSB "
           "(worst %+d) of PyTorch reference\n", NL, D, TOL, worst);
    delete dut;
    return 0;
}
