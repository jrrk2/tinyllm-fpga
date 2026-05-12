// tb_factor_ram.cpp — Verilator harness for factor_ram.sv.
//
// Exercises the read/write path: defaults after reset, host write, both
// readback (factor_rd_data) and compute-side (cur_*_factor) reflect the
// updated value.  Catches the "two-RAM split" / constant-prop issue we
// suspect Vivado is hitting on the FPGA before we burn another bitstream.

#include <verilated.h>
#include "Vfactor_ram.h"
#include <cstdio>
#include <cstdlib>

static Vfactor_ram* dut = nullptr;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

// Helper: assert a one-cycle write enable.  Data + layer + enable are
// captured at the next posedge.
static void write_swiglu_lo(uint8_t layer, uint16_t gate, uint16_t up) {
    dut->factor_wr_layer = layer & 0x1F;
    dut->factor_wr_data  = (uint32_t(up) << 16) | gate;
    dut->factor_wr_en_swiglu_lo  = 1;
    dut->factor_wr_en_swiglu_mlp = 0;
    dut->factor_wr_en_attn       = 0;
    tick();
    dut->factor_wr_en_swiglu_lo  = 0;
}

static void write_mlp(uint8_t layer, uint32_t mlp) {
    dut->factor_wr_layer = layer & 0x1F;
    dut->factor_wr_data  = mlp & 0xFFFFFF;
    dut->factor_wr_en_swiglu_lo  = 0;
    dut->factor_wr_en_swiglu_mlp = 1;
    dut->factor_wr_en_attn       = 0;
    tick();
    dut->factor_wr_en_swiglu_mlp = 0;
}

static void write_attn(uint8_t layer, uint32_t attn) {
    dut->factor_wr_layer = layer & 0x1F;
    dut->factor_wr_data  = attn & 0xFFFFFF;
    dut->factor_wr_en_swiglu_lo  = 0;
    dut->factor_wr_en_swiglu_mlp = 0;
    dut->factor_wr_en_attn       = 1;
    tick();
    dut->factor_wr_en_attn       = 0;
}

// Helper: query a factor via the readback path.  kind 0=SwiGLU lo,
// 1=mlp_out, 2=attn.  Combinational — sampled on the current cycle.
static uint32_t readback(uint8_t kind, uint8_t layer) {
    dut->factor_rd_sel = ((kind & 0x3) << 5) | (layer & 0x1F);
    dut->eval();
    return dut->factor_rd_data;
}

// Helper: read the compute-side cur_*_factor outputs for a given layer.
// These have 1-cycle registered latency from layer_idx, so set + tick.
static void read_compute(uint8_t layer,
                         uint16_t* gate, uint16_t* up,
                         uint32_t* mlp, uint32_t* attn) {
    dut->layer_idx = layer & 0x1F;
    tick();        // captures layer_idx into BRAM read addr
    tick();        // gives the registered cur_*_factor a cycle to latch
    *gate = dut->cur_sg_gate_in_factor;
    *up   = dut->cur_sg_up_in_factor;
    *mlp  = dut->cur_sg_mlp_out_factor;
    *attn = dut->cur_attn_factor;
}

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { failures++; std::fprintf(stderr, "FAIL: " fmt "\n", ##__VA_ARGS__); } \
    else         { std::fprintf(stderr, "  ok: " fmt "\n", ##__VA_ARGS__); } \
} while (0)

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vfactor_ram();

    // Reset for a few cycles to load TM_SWIGLU_SCALES defaults.
    dut->rst = 1;
    dut->factor_wr_en_swiglu_lo  = 0;
    dut->factor_wr_en_swiglu_mlp = 0;
    dut->factor_wr_en_attn       = 0;
    dut->factor_wr_layer = 0; dut->factor_wr_data = 0;
    dut->factor_rd_sel = 0;
    dut->layer_idx = 0;
    for (int i = 0; i < 5; i++) tick();
    dut->rst = 0;
    tick();

    // 1. Readback layer 0 defaults — should match generated svh.
    //    From earlier check: L0 gate=7343 up=5125 mlp=7496 attn=63.
    uint32_t sw_lo = readback(0, 0);
    uint32_t mlp   = readback(1, 0);
    uint32_t attn  = readback(2, 0);
    CHECK((sw_lo & 0xFFFF) == 7343, "L0 default gate=%u (got %u)", 7343, sw_lo & 0xFFFF);
    CHECK(((sw_lo >> 16) & 0xFFFF) == 5125, "L0 default up=%u (got %u)", 5125, (sw_lo >> 16) & 0xFFFF);
    CHECK((mlp & 0xFFFFFF) == 7496, "L0 default mlp=%u (got %u)", 7496, mlp & 0xFFFFFF);
    CHECK((attn & 0xFFFFFF) == 63,  "L0 default attn=%u (got %u)", 63,  attn & 0xFFFFFF);

    // 2. Compute-side read of layer 0 must match readback (single source of truth).
    uint16_t cg, cu;
    uint32_t cm, ca;
    read_compute(0, &cg, &cu, &cm, &ca);
    CHECK(cg == 7343, "L0 compute gate=%u (got %u)", 7343, cg);
    CHECK(cu == 5125, "L0 compute up=%u (got %u)", 5125, cu);
    CHECK(cm == 7496, "L0 compute mlp=%u (got %u)", 7496, cm);
    CHECK(ca == 63,   "L0 compute attn=%u (got %u)", 63,  ca);

    // 3. Host write: override layer 0 mlp_out_factor to 4x.  Both readback
    //    AND compute should see the new value.
    write_mlp(0, 29984);
    tick();  // let it propagate
    uint32_t mlp_after = readback(1, 0);
    CHECK((mlp_after & 0xFFFFFF) == 29984, "L0 after write mlp=29984 (readback got %u)", mlp_after & 0xFFFFFF);
    read_compute(0, &cg, &cu, &cm, &ca);
    CHECK(cm == 29984, "L0 compute after write mlp=29984 (got %u)", cm);

    // 4. Host write: zero out attn for layer 5.  Verify isolation —
    //    layer 4 unchanged, layer 5 zeroed.
    write_attn(5, 0);
    tick();
    uint32_t attn4 = readback(2, 4);
    uint32_t attn5 = readback(2, 5);
    CHECK((attn5 & 0xFFFFFF) == 0, "L5 attn zeroed (got %u)", attn5 & 0xFFFFFF);
    // L4 attn default is layer-4 calibration — just check it's NOT 0.
    CHECK((attn4 & 0xFFFFFF) != 0, "L4 attn unchanged (got %u)", attn4 & 0xFFFFFF);

    // 5. Host write: max-out gate for layer 7, then walk layer_idx through 0..NL
    //    and confirm only layer 7's compute reads return the max.
    write_swiglu_lo(7, 0xFFFF, 0xFFFF);
    tick();
    for (int li = 0; li < 30; li++) {
        read_compute(li, &cg, &cu, &cm, &ca);
        if (li == 7) {
            CHECK(cg == 0xFFFF && cu == 0xFFFF,
                  "L7 compute gate/up = 0xFFFF (got gate=%u up=%u)", cg, cu);
        } else {
            CHECK(cg != 0xFFFF || cu != 0xFFFF,
                  "L%d compute NOT overridden (got gate=%u up=%u)", li, cg, cu);
        }
    }

    std::fprintf(stderr, "\nfactor_ram: %d failures\n", failures);
    delete dut;
    return failures ? 1 : 0;
}
