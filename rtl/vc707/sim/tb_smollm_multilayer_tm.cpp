// tb_multilayer.cpp — Verilator harness for smollm_multilayer.sv (NL=3).
// Loads tm_layer_HIDDEN_IN.hex, drives the cascade, compares the final
// hidden_out against tm_layer_expected.txt.

#include "Vtb_smollm_multilayer_tm_dut.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

// D and NL can be overridden at compile time to match other configs
// (e.g. SmolLM2-real: D=576 NL=30).  The corresponding parameter
// overrides on the SV testbench come via Verilator's -pvalue+ flags.
#ifndef DUT_D
#define DUT_D 128
#endif
#ifndef DUT_NL
#define DUT_NL 3
#endif
static const int D = DUT_D;
static const int NL = DUT_NL;

static Vtb_smollm_multilayer_tm_dut* dut = nullptr;
static uint64_t cycle = 0;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
}

// 16-bit signed hex for hidden_in/expected (block-FP Q1.15 at per-layer scale).
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
    dut = new Vtb_smollm_multilayer_tm_dut;

    auto h_in = load_int16_hex("../generated/tm_layer_HIDDEN_IN.hex");
    if ((int)h_in.size() != D) {
        fprintf(stderr, "FAIL: hidden_in size %zu vs D=%d\n", h_in.size(), D);
        return 1;
    }
    auto h_ref = load_expected_signed("../generated/tm_layer_expected.txt");
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

    // Per-layer cycles scale with D × matvec_count + FFN.  At SmolLM2
    // (D=576 FFN=1536) one layer ≈ 250K cycles, so NL=30 ≈ 7.5M cycles.
    // Multiply by 4 for DDR3 stream stalls + safety margin.
    const uint64_t timeout = (uint64_t)NL * 1000000ULL;
    while (!dut->done && cycle < timeout) tick();
    if (!dut->done) {
        fprintf(stderr, "FAIL: done never asserted (cycle=%lu)\n", (unsigned long)cycle);
        delete dut;
        return 1;
    }
    printf("done=1 at cycle %lu (NL=%d layers)\n\n", (unsigned long)cycle, NL);

    // Compare final hidden_out (16-bit Q1.15 at last layer's h_out_p2 scale)
    int16_t lanes[D];
    {
        uint8_t* rb = reinterpret_cast<uint8_t*>(&dut->hidden_out);
        for (int l = 0; l < D; l++) {
            uint16_t v = rb[l*2] | (uint16_t(rb[l*2+1]) << 8);
            lanes[l] = (int16_t)v;
        }
    }

    // Dump the first 64 dut lanes verbatim so they can be diffed
    // against what the FPGA emits via selftest_verify.py.  This proves
    // hardware-vs-sim equivalence independently of the host PyTorch ref.
    printf("\nDUT-LANES (first 64, signed dec):\n");
    for (int l = 0; l < 64 && l < D; l++)
        printf("  lane %2d  %+6d   ref %+6d\n", l, lanes[l], h_ref[l]);

    // Also write all D lanes to a file so the host can decode through lm_head.
    {
        FILE* fp = std::fopen("../generated/tm_layer_dut_out.txt", "w");
        if (fp) {
            std::fprintf(fp, "# DUT hidden_out, %d lanes, Q15.9 (signed dec)\n", D);
            for (int l = 0; l < D; l++) std::fprintf(fp, "%d\n", lanes[l]);
            std::fclose(fp);
        }
    }

    int fails = 0, worst = 0;
    // 24-bit Q15.9: LSB = 1/512 ≈ 0.002 in real units.  Tolerance 512 LSB
    // = 1.0 real unit.  With matched arithmetic between sim and reference,
    // diff should be 0 — anything bigger is a bug.
    const int TOL = 512;
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
        // Histogram by chunk-of-16 to see if failures cluster
        printf("\nFailures by chunk (16 lanes each):\n");
        for (int c = 0; c < (D+15)/16; c++) {
            int chunk_fails = 0, chunk_worst = 0;
            for (int l = c*16; l < (c+1)*16 && l < D; l++) {
                int diff = (int)lanes[l] - (int)h_ref[l];
                if (std::abs(diff) > TOL) chunk_fails++;
                if (std::abs(diff) > std::abs(chunk_worst)) chunk_worst = diff;
            }
            printf("  chunk %2d: %2d/16 fail  worst %+6d\n", c, chunk_fails, chunk_worst);
        }
        delete dut;
        return 1;
    }
    printf("PASS: smollm_multilayer (NL=%d, D=%d) hidden_out within ±%d LSB "
           "(worst %+d) of PyTorch reference\n", NL, D, TOL, worst);
    delete dut;
    return 0;
}
