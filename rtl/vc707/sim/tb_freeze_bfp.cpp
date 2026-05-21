// tb_freeze_bfp.cpp — driver that DEMONSTRATES the logic-analyser freeze.
//
// Two checks, selected by env BFP_FREEZE_MODE (default "counter"):
//   counter : freeze_en=1, snap_layer_sel=L, snap_step_sel=pos.  Expect the
//             engine to halt with dbg_frozen=1 and dbg_cur_layer==L, and
//             hidden_out == golden layer-L hidden for that step.
//   cycle   : trig_cyc_en=1, trig_cyc=N.  Expect dbg_frozen=1 and
//             dbg_cur_layer == the layer executing at/after cycle N (printed;
//             a sanity bound rather than an exact golden compare).
//
// Inputs/goldens (baked by host/gen_smollm_blockfp_full.py with a per-layer
// STAGE dump — see task #8 "extract golden values"):
//   ../generated/lbfp_freeze_IN_m.hex / _e.hex     : layer-0 input (embedding)
//   ../generated/lbfp_freeze_L<NN>_m.hex / _e.hex  : golden hidden after layer NN
// Each .hex is one signed decimal per line (mantissas, then exponents file).
//
// Env: FREEZE_LAYER (default 0), FREEZE_POS (default 0), FREEZE_CYC (default 0).

#include "Vtb_freeze_bfp.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>

// Dims must match lbfp_full_cfg.svh (smollm360 default: D=960; use 576 for 135).
static constexpr int D     = 960;
static constexpr int TILE  = 16;
static constexpr int NT_D  = D / TILE;
static constexpr int MANTW = 16;
static constexpr int EXPW  = 8;

static Vtb_freeze_bfp* d;
static uint64_t cyc = 0;
static void tick() { d->clk = 0; d->eval(); d->clk = 1; d->eval(); ++cyc; }

// wr_kind read codes (mirror smollm_layer_bfp / host fpga_freeze_probe.py).
static constexpr int KIND_HOUT_M = 10, KIND_HOUT_E = 11;
static constexpr int KIND_SNAP_M = 12, KIND_SNAP_E = 13, KIND_STAGE_E = 16;

// Dummy-ethernet BRAM peek: drive the wr_kind read port on clk_wr — a SEPARATE
// clock from core `clk`, reproducing vc707_microgpt_eth's eth_clk↔core_clk
// crossing (it drives .clk_wr(eth_clk)).  Core clk is toggled alongside, so an
// engine that is NOT truly frozen would corrupt the read (the FPGA symptom).
static uint16_t read_bram(int kind, int addr, int stage = 0) {
  d->wr_kind = kind; d->wr_addr = addr; d->dbg_stage_sel = stage;
  for (int i = 0; i < 3; ++i) {            // 3 edges: regd rd_* + wr_kind_q latency
    d->clk_wr = 0; d->clk = 0; d->eval();
    d->clk_wr = 1; d->clk = 1; d->eval();
  }
  return (uint16_t)(d->wr_rdata & 0xFFFF);
}
static inline int rd_s16(uint16_t v) { return (v & 0x8000) ? (int)v - 0x10000 : (int)v; }
static inline int rd_s8 (uint16_t v) { v &= 0xFF; return (v & 0x80) ? (int)v - 0x100 : (int)v; }

static std::vector<int> load_ints(const std::string& p, bool required) {
  std::vector<int> v; std::ifstream f(p);
  if (!f) { if (required) { std::fprintf(stderr, "ERROR: missing %s\n", p.c_str()); std::exit(1); }
            return v; }
  int x; while (f >> x) v.push_back(x);
  return v;
}

// Pack signed mantissas (MANTW bits) / exponents (EXPW bits) into a Verilator
// wide signal (WData = uint32_t words, little-endian bit order).
template <typename WIDE>
static void pack(WIDE& dst, const std::vector<int>& vals, int width, int n, int words) {
  for (int w = 0; w < words; ++w) dst[w] = 0;
  for (int i = 0; i < n && i < (int)vals.size(); ++i) {
    uint32_t bits = (uint32_t)(vals[i]) & ((width >= 32) ? 0xFFFFFFFFu : ((1u << width) - 1));
    long lo = (long)i * width;
    for (int b = 0; b < width; ++b) {
      if (bits & (1u << b)) dst[(lo + b) >> 5] |= (1u << ((lo + b) & 31));
    }
  }
}

template <typename WIDE>
static int get_signed(const WIDE& src, int idx, int width) {
  long lo = (long)idx * width; uint32_t bits = 0;
  for (int b = 0; b < width; ++b)
    if (src[(lo + b) >> 5] & (1u << ((lo + b) & 31))) bits |= (1u << b);
  int v = (int)bits;
  if (bits & (1u << (width - 1))) v -= (1 << width);   // sign-extend
  return v;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  d = new Vtb_freeze_bfp;

  const int  L   = std::getenv("FREEZE_LAYER") ? atoi(std::getenv("FREEZE_LAYER")) : 0;
  const int  POS = std::getenv("FREEZE_POS")   ? atoi(std::getenv("FREEZE_POS"))   : 0;
  const long CYC = std::getenv("FREEZE_CYC")   ? atol(std::getenv("FREEZE_CYC"))   : 0;
  const std::string mode = std::getenv("BFP_FREEZE_MODE") ? std::getenv("BFP_FREEZE_MODE") : "counter";

  // Optional golden input vector.  Absent → zeros, which still exercises the
  // freeze CONTROL (the FSM advances through layers regardless of data values).
  auto in_m  = load_ints("../generated/lbfp_freeze_IN_m.hex", false);
  auto in_e  = load_ints("../generated/lbfp_freeze_IN_e.hex", false);
  if (in_m.empty()) std::fprintf(stderr, "[note] no golden input — feeding zeros (control-only demo)\n");
  char gp[64]; std::snprintf(gp, sizeof(gp), "../generated/lbfp_freeze_L%02d_m.hex", L);
  auto gold_m = load_ints(gp, false);   // optional golden for compare

  // Reset.
  d->clk = 0; d->rst = 1; d->start = 0;
  d->clk_wr = 0; d->wr_kind = 0; d->wr_addr = 0; d->dbg_stage_sel = 0;
  d->freeze_en = 0; d->trig_cyc_en = 0; d->trig_cyc = 0;
  d->snap_layer_sel = 0; d->snap_step_sel = 0; d->pos = 0; d->kv_pos = 0;
  pack(d->hidden_in_m, in_m, MANTW, D,    (D*MANTW + 31)/32);
  pack(d->hidden_in_e, in_e, EXPW,  NT_D, (NT_D*EXPW + 31)/32);
  d->eval();
  for (int i = 0; i < 8; ++i) tick();
  d->rst = 0; tick();

  // Program the freeze.
  d->pos = POS; d->kv_pos = POS;
  if (mode == "cycle") { d->trig_cyc_en = 1; d->trig_cyc = (uint32_t)CYC; }
  else                 { d->freeze_en = 1; d->snap_layer_sel = (uint8_t)L; d->snap_step_sel = POS; }

  // Run.
  d->start = 1; tick(); d->start = 0;
  const uint64_t budget = 5'000'000;
  while (!d->done && cyc < budget) tick();

  // Max per-tile exponent of hidden_out — the saturation indicator (matches
  // the FPGA probe's max_exp; e ~ +120 means a tile's BFP exponent blew up).
  int max_exp = -128; double max_abs = 0;
  for (int t = 0; t < NT_D; ++t) { int e = get_signed(d->hidden_out_e, t, EXPW); if (e > max_exp) max_exp = e; }
  for (int i = 0; i < D; ++i) { double v = std::fabs((double)get_signed(d->hidden_out_m, i, MANTW) * std::ldexp(1.0, get_signed(d->hidden_out_e, i/TILE, EXPW) - 15)); if (v > max_abs) max_abs = v; }
  std::printf("mode=%s  done=%d  dbg_frozen=%d  dbg_cur_layer=%d  dbg_cyc=%u  weight_hash=0x%08x  max_exp=%d  max|val|=%.4g  (host cyc=%llu)\n",
              mode.c_str(), d->done, d->dbg_frozen, d->dbg_cur_layer, d->dbg_cyc, d->weight_hash, max_exp, max_abs,
              (unsigned long long)cyc);

  int rc = 0;
  if (!d->dbg_frozen) { std::printf("FAIL: engine did not freeze\n"); rc = 1; }
  if (mode == "counter" && d->dbg_cur_layer != L) {
    std::printf("FAIL: froze at layer %d, expected %d\n", d->dbg_cur_layer, L); rc = 1;
  }
  // Diagnostic: confirm the input is actually being fed (decoded), and show
  // output vs golden — decode BFP m * 2^(e-15) per tile (NOT raw mantissas).
  std::snprintf(gp, sizeof(gp), "../generated/lbfp_freeze_L%02d_e.hex", L);
  auto gold_e = load_ints(gp, false);
  auto dec = [](int m, int e) { return (double)m * std::ldexp(1.0, e - 15); };
  std::printf("input[0..3] decoded:");
  for (int i = 0; i < 4; ++i)
    std::printf(" %.5f", dec(in_m.empty()?0:in_m[i], in_e.empty()?0:in_e[i/TILE]));
  std::printf("\n");
  if (!gold_m.empty() && !gold_e.empty()) {
    double worst = 0; int fails = 0;
    for (int i = 0; i < D; ++i) {
      double got = dec(get_signed(d->hidden_out_m, i, MANTW), get_signed(d->hidden_out_e, i/TILE, EXPW));
      double ref = dec(gold_m[i], gold_e[i/TILE]);
      double diff = got - ref;
      if (std::abs(diff) > std::abs(worst)) worst = diff;
      if (std::abs(diff) > 1e-3 * (std::abs(ref) + 1e-6)) ++fails;   // 0.1% rel tol
    }
    std::printf("out[0..3] decoded:  ");
    for (int i = 0; i < 4; ++i)
      std::printf(" %.5f", dec(get_signed(d->hidden_out_m, i, MANTW), get_signed(d->hidden_out_e, i/TILE, EXPW)));
    std::printf("\ngold[0..3] decoded: ");
    for (int i = 0; i < 4; ++i) std::printf(" %.5f", dec(gold_m[i], gold_e[i/TILE]));
    std::printf("\nhidden_out vs golden L%02d (decoded, 0.1%% tol): %d/%d differ, worst %+.4f\n",
                L, fails, D, worst);
    if (fails) rc = 1;
  } else {
    std::printf("(no golden L%02d — out[0..7] mant:", L);
    for (int i = 0; i < 8; ++i) std::printf(" %d", get_signed(d->hidden_out_m, i, MANTW));
    std::printf(")\n");
  }
  // ---- Dummy-ethernet read-path validation -------------------------------
  // Read hout back through the wr_kind 10/11 peek (the host's exact path, on a
  // SEPARATE clk_wr) and compare to the directly-wired hidden_out (ground
  // truth).  Repeat 3× to check the read is DETERMINISTIC — on the FPGA three
  // identical freezes gave three different houts.  A mismatch here localises
  // the bug to the read datapath / freeze capture; a clean match exonerates
  // them (→ the FPGA non-determinism is in eth/UDP or autoregress re-trigger).
  if (d->dbg_frozen) {
    // Check both read paths against ground truth (the frozen hidden_out):
    //   live  = wr_kind 10/11 (hout_m/hout_e)  — overwritten every layer/token,
    //           races with a running engine on the real chip.
    //   snap  = wr_kind 12/13 (snap_m/snap_e)  — latched ONCE at the matching
    //           (snap_layer_sel, snap_step_sel), never overwritten until rst.
    // The snapshot is the stable read (option b): it must match ground truth.
    for (int which = 0; which < 2; ++which) {
      const int KM = which ? KIND_SNAP_M : KIND_HOUT_M;
      const int KE = which ? KIND_SNAP_E : KIND_HOUT_E;
      const char* nm = which ? "snap(12/13)" : "live(10/11)";
      int mism_m = 0, mism_e = 0, nondet = 0, probe0[D];
      for (int pass = 0; pass < 3; ++pass) {
        for (int t = 0; t < NT_D; ++t) {
          int pe = rd_s8(read_bram(KE, t));
          if (pass == 0 && pe != get_signed(d->hidden_out_e, t, EXPW)) ++mism_e;
        }
        for (int i = 0; i < D; ++i) {
          int pm = rd_s16(read_bram(KM, i));
          if (pass == 0) { probe0[i] = pm; if (pm != get_signed(d->hidden_out_m, i, MANTW)) ++mism_m; }
          else if (pm != probe0[i]) ++nondet;
        }
      }
      std::printf("read-path %-11s: hout_m mism=%d/%d  hout_e mism=%d/%d  nondet=%d/%d  -> %s\n",
                  nm, mism_m, D, mism_e, NT_D, nondet, 2 * D,
                  (mism_m || mism_e || nondet) ? "MISMATCH" : "OK");
      if (which == 1 && (mism_m || mism_e || nondet)) rc = 1;   // snapshot must be clean
    }
  }

  std::printf("%s\n", rc ? "FREEZE DEMO: FAIL" : "FREEZE DEMO: PASS");
  delete d; return rc;
}
