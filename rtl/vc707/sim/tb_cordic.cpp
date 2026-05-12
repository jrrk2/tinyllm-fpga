// tb_cordic.cpp — exhaustive accuracy test for cordic_sincos.sv.
//
// Sweeps 10001 angles uniformly across [-π, π), drives each through the
// CORDIC, and compares the (cos, sin) output to the numpy/std-math
// reference quantised to Q1.15.  Reports max |diff| in LSBs and the
// histogram of error magnitudes.

#include "Vcordic_sincos.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>

static Vcordic_sincos* dut = nullptr;
static uint64_t cycle = 0;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
}

static int16_t sat_q15_from_double(double v) {
    long q = (long)std::lround(v * 32768.0);
    if (q >  32767) q = 32767;
    if (q < -32768) q = -32768;
    return (int16_t)q;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vcordic_sincos;

    // Reset
    dut->rst = 1;
    dut->start = 0;
    dut->angle_in = 0;
    for (int i = 0; i < 4; i++) tick();
    dut->rst = 0;
    tick();

    const int N = 10001;
    int worst_cos = 0, worst_sin = 0;
    int hist_cos[16] = {0}, hist_sin[16] = {0};   // |diff| 0,1,2,...,15+
    double worst_cos_angle = 0.0, worst_sin_angle = 0.0;

    const double SCALE = (double)(1LL << 27);    // Q3.27
    for (int i = 0; i < N; i++) {
        // Angle uniformly in [-π, π).
        double angle_rad = -M_PI + (2.0 * M_PI * i) / N;
        long long ang_q = (long long)std::llround(angle_rad * SCALE);
        // Q3.27 fits in signed 31-bit; clamp.
        if (ang_q >  ((1LL << 30) - 1)) ang_q =  (1LL << 30) - 1;
        if (ang_q < -(1LL << 30))       ang_q = -(1LL << 30);

        dut->angle_in = (uint32_t)(ang_q & 0x7FFFFFFFLL);   // low 31 bits
        dut->start    = 1;
        tick();
        dut->start    = 0;

        // Wait for valid (1 + 16 + 1 = ~18 cycles)
        uint64_t t0 = cycle;
        while (!dut->valid && (cycle - t0) < 64) tick();
        if (!dut->valid) {
            fprintf(stderr, "FAIL: valid never asserted at i=%d (cycle=%lu)\n",
                    i, (unsigned long)cycle);
            delete dut;
            return 1;
        }

        int16_t dut_cos = (int16_t)dut->cos_out;
        int16_t dut_sin = (int16_t)dut->sin_out;
        // Compare against the *quantised* angle (what the engine actually
        // sees in Q3.27) so we isolate CORDIC iteration error from input
        // quantisation noise.
        double angle_q = (double)ang_q / SCALE;
        int16_t ref_cos = sat_q15_from_double(std::cos(angle_q));
        int16_t ref_sin = sat_q15_from_double(std::sin(angle_q));

        int dc = (int)dut_cos - (int)ref_cos;
        int ds = (int)dut_sin - (int)ref_sin;
        int adc = std::abs(dc), ads = std::abs(ds);
        if (adc > std::abs(worst_cos)) { worst_cos = dc; worst_cos_angle = angle_rad; }
        if (ads > std::abs(worst_sin)) { worst_sin = ds; worst_sin_angle = angle_rad; }

        hist_cos[std::min(adc, 15)]++;
        hist_sin[std::min(ads, 15)]++;

        // Drain done pulse
        tick();
    }

    printf("CORDIC accuracy over %d angles in [-π, π):\n", N);
    printf("  cos  worst diff = %+d LSB at angle %.6f rad (cos=%.6f)\n",
           worst_cos, worst_cos_angle, std::cos(worst_cos_angle));
    printf("  sin  worst diff = %+d LSB at angle %.6f rad (sin=%.6f)\n",
           worst_sin, worst_sin_angle, std::sin(worst_sin_angle));
    printf("\n  |diff|  cos count  sin count\n");
    int cum_cos = 0, cum_sin = 0;
    for (int b = 0; b < 16; b++) {
        cum_cos += hist_cos[b]; cum_sin += hist_sin[b];
        if (b == 15)
            printf("   ≥%2d   %8d (%4.1f%%)  %8d (%4.1f%%)\n",
                   b, hist_cos[b], 100.0*hist_cos[b]/N,
                      hist_sin[b], 100.0*hist_sin[b]/N);
        else
            printf("   %3d   %8d (%4.1f%%)  %8d (%4.1f%%)\n",
                   b, hist_cos[b], 100.0*hist_cos[b]/N,
                      hist_sin[b], 100.0*hist_sin[b]/N);
    }

    int worst = std::max(std::abs(worst_cos), std::abs(worst_sin));
    if (worst <= 2) {
        printf("\nPASS: CORDIC within ±%d LSB across all %d angles\n", worst, N);
        delete dut;
        return 0;
    }
    printf("\nFAIL: worst |diff| = %d LSB (>2)\n", worst);
    delete dut;
    return 1;
}
