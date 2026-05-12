// tb_persist.cpp — long-running Verilator backend for matvec_int8_engine.
//
// Reads request frames on stdin, drives matvec_int8_engine, writes responses
// on stdout.  One process per Python session; weights are sent per Linear.
//
// Protocol:
//   request  : "LIN0" | u32 in_dim | u32 out_dim
//                     | int8  w_int8[out_dim * in_dim]   (row-major, lane stride = in_dim)
//                     | int16 scale_q15[out_dim]
//                     | int16 x_q15[in_dim]
//   response : int16  y_q15[out_dim]
//   request  : "QUIT" → exit cleanly
//
// out_dim must be a multiple of LANES=16.  All in_dim are multiples of 8.

#include "Vmatvec_int8_engine.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

static const int LANES = 16;
static Vmatvec_int8_engine* dut = nullptr;

static inline void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

static int read_exact(void* buf, size_t n) {
    char* p = (char*)buf;
    while (n) {
        ssize_t r = ::read(0, p, n);
        if (r <= 0) return -1;
        p += r; n -= r;
    }
    return 0;
}

static int write_exact(const void* buf, size_t n) {
    const char* p = (const char*)buf;
    while (n) {
        ssize_t r = ::write(1, p, n);
        if (r <= 0) return -1;
        p += r; n -= r;
    }
    return 0;
}

// Run one 16-output chunk through the engine.
//   w     : [LANES * in_dim] INT8, lane stride = in_dim (lane l, k = w[l*in_dim + k])
//   scale : [LANES] int16  (Q1.15 per-lane scale)
//   x     : [in_dim] int16
//   y     : [LANES] int16  (saturated Q1.15)
static void matvec16(int in_dim,
                     const int8_t*  w,
                     const int16_t* scale,
                     const int16_t* x,
                     int16_t*       y) {
    // Reset accumulator
    dut->in_valid = 0;
    dut->in_last  = 0;
    dut->scale_valid = 0;
    dut->acc_clear = 1;
    tick();
    dut->acc_clear = 0;

    // Stream in_dim cycles
    uint8_t* wb = reinterpret_cast<uint8_t*>(&dut->w_int8);
    for (int k = 0; k < in_dim; k++) {
        dut->in_value = (uint16_t)x[k];
        dut->in_valid = 1;
        dut->in_last  = (k == in_dim - 1);
        for (int l = 0; l < LANES; l++)
            wb[l] = (uint8_t)w[l * in_dim + k];
        tick();
    }
    dut->in_valid = 0;
    dut->in_last  = 0;
    dut->in_value = 0;

    // Single-cycle MAC: one drain tick is enough (no pipeline registers).
    tick();

    // Apply scale, capture output
    uint8_t* sb = reinterpret_cast<uint8_t*>(&dut->scale_q15);
    for (int l = 0; l < LANES; l++) {
        uint16_t v = (uint16_t)scale[l];
        sb[l*2+0] = v & 0xFF;
        sb[l*2+1] = v >> 8;
    }
    dut->scale_valid = 1;
    tick();
    dut->scale_valid = 0;
    tick();   // out_valid pulses; output settles

    uint8_t* ob = reinterpret_cast<uint8_t*>(&dut->out_value);
    for (int l = 0; l < LANES; l++) {
        uint16_t v = ob[l*2] | (uint16_t(ob[l*2+1]) << 8);
        y[l] = (int16_t)v;
    }
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vmatvec_int8_engine;
    dut->rst = 1;
    dut->in_value = 0; dut->in_valid = 0; dut->in_last = 0;
    dut->scale_valid = 0; dut->acc_clear = 0;
    memset(&dut->w_int8,    0, sizeof(dut->w_int8));
    memset(&dut->scale_q15, 0, sizeof(dut->scale_q15));
    for (int i = 0; i < 4; i++) tick();
    dut->rst = 0;
    tick();

    char magic[4];
    while (read_exact(magic, 4) == 0) {
        if (memcmp(magic, "LIN0", 4) == 0) {
            uint32_t in_dim, out_dim;
            if (read_exact(&in_dim,  4) < 0) break;
            if (read_exact(&out_dim, 4) < 0) break;
            std::vector<int8_t>  w   (size_t(out_dim) * in_dim);
            std::vector<int16_t> scl (out_dim);
            std::vector<int16_t> x   (in_dim);
            std::vector<int16_t> y   (out_dim);
            if (read_exact(w.data(),   size_t(out_dim) * in_dim) < 0) break;
            if (read_exact(scl.data(), size_t(out_dim) * 2)      < 0) break;
            if (read_exact(x.data(),   size_t(in_dim)  * 2)      < 0) break;

            for (uint32_t r = 0; r < out_dim; r += LANES) {
                int16_t lane_out[LANES];
                matvec16(in_dim,
                         w.data()   + size_t(r) * in_dim,
                         scl.data() + r,
                         x.data(),
                         lane_out);
                for (int l = 0; l < LANES; l++) y[r + l] = lane_out[l];
            }
            if (write_exact(y.data(), size_t(out_dim) * 2) < 0) break;
        } else if (memcmp(magic, "QUIT", 4) == 0) {
            break;
        } else {
            fprintf(stderr, "tb_persist: unknown magic %02x%02x%02x%02x\n",
                    magic[0], magic[1], magic[2], magic[3]);
            break;
        }
    }

    delete dut;
    return 0;
}
