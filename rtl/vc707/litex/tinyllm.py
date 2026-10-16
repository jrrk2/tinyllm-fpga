#!/usr/bin/env python3
"""tinyllm as a LiteX peripheral.

Wraps autoregress_bfp_top (the SmolLM2 block-FP autoregressive engine) as a
LiteXModule: CSRs for everything the host regmap used to carry, an AXI4 read
master that the SoC attaches to a LiteDRAM port, and the clock-domain
crossings the old vc707_microgpt_eth top did by hand.

Clock domains
  core  : the engine's compute clock (40 MHz, an extra MMCM output).
  sys   : CSRs, the BRAM write port (clk_wr) and the AXI side (clk_axi).

Register semantics follow host/gen_regmap.py so the firmware is a mechanical
port of fw_engine.c:
  restart      write -> one restart pulse (resets the engine, clears
               done_latched / has_run / the first-AR,R snapshot)
  n_prompt     active prompt length (1..NPROMPT_MAX)
  n_gen        generated tokens per run, 0 = NGEN; clamped to the context
  status       [0] done (live) [1] done_latched [2] has_run
               [3] first_ar_seen [4] first_r_seen [5] frozen
  bram_target  {inc[31], kind[22:18], addr[17:0]}  (old 0x060)
  bram_data    [15:0] -> one write pulse at the target, auto-inc  (old 0x061)
  bram_rdata   [15:0] BRAM[kind, addr] read-back                  (old 0x062)
  result       N_STEPS x 16-bit result tokens (old 0x1D0 window)
  ws_base_*    DDR3 byte offsets of the 18 weight regions; defaults are the
               lbfp_ddr3.svh layout for this model, firmware may override
               (e.g. from a setup file on the SD card).
"""

import os

from migen import *
from migen.genlib.cdc import MultiReg, PulseSynchronizer

from litex.gen import LiteXModule
from litex.soc.interconnect.csr import CSR, CSRStatus, CSRStorage, CSRField
from litex.soc.interconnect import axi

# ----------------------------------------------------------------------------
# Model configuration (rtl/generated/lbfp_full_cfg.svh) and DDR3 region map
# (src/smollm/lbfp_ddr3.svh), evaluated in Python.
# ----------------------------------------------------------------------------

CFG_KEYS = ["D", "HQ", "HKV", "HD", "FFN", "NL", "MAX_CTX", "VOCAB",
            "NPROMPT", "NPROMPT_MAX", "NGEN"]

def read_cfg(path):
    cfg = {}
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) >= 3 and t[0] == "`define" and t[1].startswith("LBFP_FULL_"):
                cfg[t[1][len("LBFP_FULL_"):]] = int(t[2])
    missing = [k for k in CFG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    return cfg

REGION_ORDER = ["WQ_m", "WQ_e", "WK_m", "WK_e", "WV_m", "WV_e", "WO_m", "WO_e",
                "WG_m", "WG_e", "WU_m", "WU_e", "WDN_m", "WDN_e",
                "EMBED_m", "EMBED_e", "EMBED_LU_m", "EMBED_LU_e"]

def region_bases(cfg):
    """Port of lbfp_ddr3.svh: byte offsets of every region, in DDR3 order."""
    BYTES_M_PER_COL, BYTES_E_PER_TILE, ALIGN = 32, 16, 4096
    NL, D, HKV, HD, FFN, VOCAB = (cfg[k] for k in ("NL", "D", "HKV", "HD", "FFN", "VOCAB"))
    a4k = lambda n: (n + ALIGN - 1) & ~(ALIGN - 1)
    size_m = lambda dout, din: a4k(NL * (dout // 16) * din * BYTES_M_PER_COL)
    size_e = lambda dout, din: a4k(NL * (dout // 16) * (din // 16) * BYTES_E_PER_TILE)
    gamma_m, gamma_e = a4k(NL * D * 2), a4k(NL * (D // 16))
    b = {}
    b["WQ_m"]  = 0
    b["WQ_e"]  = b["WQ_m"]  + size_m(D, D)
    b["WK_m"]  = b["WQ_e"]  + size_e(D, D)
    b["WK_e"]  = b["WK_m"]  + size_m(HKV * HD, D)
    b["WV_m"]  = b["WK_e"]  + size_e(HKV * HD, D)
    b["WV_e"]  = b["WV_m"]  + size_m(HKV * HD, D)
    b["WO_m"]  = b["WV_e"]  + size_e(HKV * HD, D)
    b["WO_e"]  = b["WO_m"]  + size_m(D, D)
    b["WG_m"]  = b["WO_e"]  + size_e(D, D)
    b["WG_e"]  = b["WG_m"]  + size_m(FFN, D)
    b["WU_m"]  = b["WG_e"]  + size_e(FFN, D)
    b["WU_e"]  = b["WU_m"]  + size_m(FFN, D)
    b["WDN_m"] = b["WU_e"]  + size_e(FFN, D)
    b["WDN_e"] = b["WDN_m"] + size_m(D, FFN)
    b["G1_m"]  = b["WDN_e"] + size_e(D, FFN)
    b["G1_e"]  = b["G1_m"]  + gamma_m
    b["G2_m"]  = b["G1_e"]  + gamma_e
    b["G2_e"]  = b["G2_m"]  + gamma_m
    b["EMBED_m"]    = b["G2_e"]       + gamma_e
    b["EMBED_e"]    = b["EMBED_m"]    + VOCAB // 16 * D * BYTES_M_PER_COL
    b["EMBED_LU_m"] = b["EMBED_e"]    + VOCAB // 16 * (D // 16) * BYTES_E_PER_TILE
    b["EMBED_LU_e"] = b["EMBED_LU_m"] + VOCAB * D * 2
    b["NORM_W_m"]   = b["EMBED_LU_e"] + VOCAB * 64
    b["NORM_W_e"]   = b["NORM_W_m"]   + a4k(D * 2)
    b["END"]        = b["NORM_W_e"]   + a4k(D // 16)
    return b

# Engine sources, relative to rtl/vc707/src/smollm (order as scripts/run.tcl).
ENGINE_SOURCES = [
    "cordic_sincos.sv", "matvec_bfp_engine.sv", "rmsnorm_bfp.sv", "rope_bfp.sv",
    "swiglu_bfp.sv", "softmax_bfp.sv", "residual_bfp.sv", "bfp_sdpram.sv",
    "bfp_sdpram_striped.sv", "bfp_sdpram_packed.sv", "bfp_sdpram_packed_pr.sv",
    "weight_streamer_bfp_mt.sv", "smollm_layer_bfp.sv", "embed_lookup_bfp.sv",
    "smollm_multilayer_tm_bfp.sv", "smollm_decode_head_bfp.sv", "axi_r_slice.sv",
    "autoregress_bfp_top.sv",
]

class TinyLLM(LiteXModule):
    def __init__(self, platform, cfg, smollm_dir, generated_dir, build_version=0,
                 prefix="lbfp_full_", core_cd="tinyllm_core", axi_cd="sys",
                 axi_addr_width=30, axi_id_width=5):
        self.cfg   = cfg
        self.bases = region_bases(cfg)
        N_STEPS    = cfg["NPROMPT_MAX"] + cfg["NGEN"]
        NPA_W      = bits_for(cfg["NPROMPT_MAX"])

        # -- Sources -----------------------------------------------------
        for f in ENGINE_SOURCES:
            platform.add_source(os.path.join(smollm_dir, f), language="systemverilog")
        platform.add_verilog_include_path(smollm_dir)
        platform.add_verilog_include_path(generated_dir)

        # -- CSRs (sys) ----------------------------------------------------
        self.restart     = CSR()
        self.arm         = CSR()
        self.n_prompt    = CSRStorage(NPA_W, reset=cfg["NPROMPT"],
                                      description="active prompt length (1..NPROMPT_MAX)")
        self.status      = CSRStatus(fields=[
            CSRField("done",          1, description="engine done (live, 1 sys cycle)"),
            CSRField("done_latched",  1, description="done since the last restart"),
            CSRField("has_run",       1, description="single-shot run consumed"),
            CSRField("first_ar_seen", 1, description="an AR was accepted since restart/arm"),
            CSRField("first_r_seen",  1, description="an R beat was accepted since restart/arm"),
            CSRField("frozen",        1, description="engine frozen at the trigger"),
        ])
        self.bram_target = CSRStorage(32, description="{inc[31], kind[22:18], addr[17:0]}")
        self.bram_data   = CSRStorage(16, description="write -> one BRAM write at the target")
        self.bram_rdata  = CSRStatus(16,  description="BRAM[kind, addr] read-back")
        self.result      = CSRStatus(N_STEPS * 16, description="result tokens, 16 bit each")
        self.weight_hash = CSRStatus(32)
        self.build_version = CSRStatus(32, reset=build_version)
        self.dbg_first_araddr = CSRStatus(axi_addr_width)
        self.dbg_first_rdata  = CSRStatus(512)
        self.snap_layer_sel = CSRStorage(5)
        self.snap_step_sel  = CSRStorage(11)
        self.dbg_stage_sel  = CSRStorage(5)
        self.freeze_en      = CSRStorage(1)
        self.trig_cyc_en    = CSRStorage(1)
        self.trig_cyc       = CSRStorage(32)
        self.dbg_cyc        = CSRStatus(32)
        self.dbg_cur_layer  = CSRStatus(5)
        self.model = CSRStatus(fields=[
            CSRField("n_steps",     8, reset=N_STEPS),
            CSRField("nprompt_max", 8, reset=cfg["NPROMPT_MAX"]),
            CSRField("ngen",        8, reset=cfg["NGEN"]),
            CSRField("nl",          8, reset=cfg["NL"]),
        ])
        self.n_gen = CSRStorage(bits_for(N_STEPS),
                                description="generated tokens per run; 0 = NGEN.  The engine "
                                            "clamps n_prompt + n_gen to min(N_STEPS, MAX_CTX)")
        ws_base = {}
        for r in REGION_ORDER:
            csr = CSRStorage(axi_addr_width, reset=self.bases[r],
                             name=f"ws_base_{r}", description=f"DDR3 byte offset of {r}")
            setattr(self, f"ws_base_{r}", csr)
            ws_base[r] = csr

        # -- AXI read master (sys) ----------------------------------------
        self.axi = axi.AXIInterface(data_width=512, address_width=axi_addr_width,
                                    id_width=axi_id_width, clock_domain=axi_cd)
        self.comb += [self.axi.aw.valid.eq(0), self.axi.w.valid.eq(0), self.axi.b.ready.eq(1)]

        # -- Restart / start / done -----------------------------------------
        # restart: sys pulse -> toggle -> 3FF into core -> edge = 1-cycle pulse.
        restart_tog  = Signal()
        self.sync += If(self.restart.re, restart_tog.eq(~restart_tog))
        tog_core = Signal(3)
        sync_core = getattr(self.sync, core_cd)
        sync_core += tog_core.eq(Cat(restart_tog, tog_core[:2]))
        restart_core = Signal()
        self.comb += restart_core.eq(tog_core[2] ^ tog_core[1])
        rst_core = Signal()
        self.comb += rst_core.eq(ResetSignal(core_cd) | restart_core)

        done_core   = Signal()
        has_run_c   = Signal(reset=1)   # wait for the first host restart
        start_c     = Signal()
        sync_core += [
            If(restart_core,   has_run_c.eq(0), start_c.eq(0))
            .Elif(done_core,   has_run_c.eq(1), start_c.eq(0))
            .Elif(~has_run_c,  start_c.eq(1))
            .Else(             start_c.eq(0)),
        ]
        self.done_ps = PulseSynchronizer(core_cd, axi_cd)
        self.comb += self.done_ps.i.eq(done_core)
        done_latched = Signal(); has_run_s = Signal()
        self.sync += [
            If(self.restart.re, done_latched.eq(0), has_run_s.eq(0))
            .Elif(self.done_ps.o, done_latched.eq(1), has_run_s.eq(1)),
        ]

        # -- quasi-static CDC ---------------------------------------------
        n_prompt_c = Signal(NPA_W); self.specials += MultiReg(self.n_prompt.storage, n_prompt_c, core_cd)
        n_gen_c = Signal(bits_for(N_STEPS)); self.specials += MultiReg(self.n_gen.storage, n_gen_c, core_cd)
        snap_layer_c = Signal(5);   self.specials += MultiReg(self.snap_layer_sel.storage, snap_layer_c, core_cd)
        snap_step_c  = Signal(11);  self.specials += MultiReg(self.snap_step_sel.storage,  snap_step_c,  core_cd)
        stage_sel_c  = Signal(5);   self.specials += MultiReg(self.dbg_stage_sel.storage,  stage_sel_c,  core_cd)
        freeze_c     = Signal();    self.specials += MultiReg(self.freeze_en.storage,      freeze_c,     core_cd)
        trig_en_c    = Signal();    self.specials += MultiReg(self.trig_cyc_en.storage,    trig_en_c,    core_cd)
        trig_cyc_c   = Signal(32);  self.specials += MultiReg(self.trig_cyc.storage,       trig_cyc_c,   core_cd)
        result_c = Signal(N_STEPS * 16); self.specials += MultiReg(result_c, self.result.status)
        whash_c  = Signal(32);           self.specials += MultiReg(whash_c,  self.weight_hash.status)
        dbg_cyc_c = Signal(32);          self.specials += MultiReg(dbg_cyc_c, self.dbg_cyc.status)
        cur_layer_c = Signal(5);         self.specials += MultiReg(cur_layer_c, self.dbg_cur_layer.status)
        frozen_c = Signal(); frozen_s = Signal(); self.specials += MultiReg(frozen_c, frozen_s)

        # -- BRAM write port (sys) -----------------------------------------
        wr_en = Signal(); wr_addr = Signal(18); wr_data = Signal(16); wr_kind = Signal(5)
        bram_addr = Signal(18); bram_rdata = Signal(16)
        self.sync += [
            wr_en.eq(0),
            If(self.bram_target.re,
                bram_addr.eq(self.bram_target.storage[0:18]),
                wr_kind.eq(self.bram_target.storage[18:23]),
            ),
            If(self.bram_data.re,
                wr_data.eq(self.bram_data.storage[0:16]),
                wr_addr.eq(bram_addr),
                wr_en.eq(1),
                If(self.bram_target.storage[31], bram_addr.eq(bram_addr + 1)),
            ),
            self.bram_rdata.status.eq(bram_rdata),
        ]
        port_addr = Signal(18)
        self.comb += port_addr.eq(Mux(wr_en, wr_addr, bram_addr))

        # -- first AR / R snapshot after restart|arm (sys) ------------------
        ar_seen = Signal(); r_seen = Signal()
        self.sync += [
            If(self.restart.re | self.arm.re, ar_seen.eq(0), r_seen.eq(0))
            .Else(
                If(self.axi.ar.valid & self.axi.ar.ready & ~ar_seen,
                    ar_seen.eq(1), self.dbg_first_araddr.status.eq(self.axi.ar.addr)),
                If(self.axi.r.valid & self.axi.r.ready & ~r_seen,
                    r_seen.eq(1), self.dbg_first_rdata.status.eq(self.axi.r.data)),
            ),
        ]
        self.comb += [
            self.status.fields.done.eq(self.done_ps.o),
            self.status.fields.done_latched.eq(done_latched),
            self.status.fields.has_run.eq(has_run_s),
            self.status.fields.first_ar_seen.eq(ar_seen),
            self.status.fields.first_r_seen.eq(r_seen),
            self.status.fields.frozen.eq(frozen_s),
        ]

        # -- The engine ----------------------------------------------------
        ws_ports = {f"i_ws_base_{r}": ws_base[r].storage for r in REGION_ORDER}
        self.specials += Instance("autoregress_bfp_top",
            p_D=cfg["D"], p_H_Q=cfg["HQ"], p_H_KV=cfg["HKV"], p_HD=cfg["HD"],
            p_FFN=cfg["FFN"], p_NL=cfg["NL"], p_MAX_CTX=cfg["MAX_CTX"], p_VOCAB=cfg["VOCAB"],
            p_NPROMPT_MAX=cfg["NPROMPT_MAX"], p_N_PROMPT=cfg["NPROMPT"], p_N_GEN=cfg["NGEN"],
            p_PREFIX=prefix, p_STREAM_LOOKUP=1,
            p_AXI_ADDR_WIDTH=axi_addr_width, p_AXI_ID_WIDTH=axi_id_width,
            i_clk=ClockSignal(core_cd), i_rst=rst_core, i_start=start_c,
            i_n_prompt_active=n_prompt_c, i_n_gen_req=n_gen_c, o_done=done_core,
            o_result_tokens=result_c, o_weight_hash=whash_c,
            i_wr_kind=wr_kind, i_wr_addr=port_addr, i_wr_data=wr_data, i_wr_en=wr_en,
            i_clk_wr=ClockSignal(axi_cd), o_wr_rdata=bram_rdata,
            i_snap_layer_sel=snap_layer_c, i_snap_step_sel=snap_step_c,
            i_dbg_stage_sel=stage_sel_c, i_freeze_en=freeze_c,
            i_trig_cyc_en=trig_en_c, i_trig_cyc=trig_cyc_c,
            o_dbg_cyc=dbg_cyc_c, o_dbg_cur_layer=cur_layer_c, o_dbg_frozen=frozen_c,
            i_clk_axi=ClockSignal(axi_cd), i_rst_axi=ResetSignal(axi_cd),
            o_m_axi_arvalid=self.axi.ar.valid, i_m_axi_arready=self.axi.ar.ready,
            o_m_axi_arid=self.axi.ar.id, o_m_axi_araddr=self.axi.ar.addr,
            o_m_axi_arlen=self.axi.ar.len, o_m_axi_arsize=self.axi.ar.size,
            o_m_axi_arburst=self.axi.ar.burst, o_m_axi_arlock=self.axi.ar.lock,
            o_m_axi_arcache=self.axi.ar.cache, o_m_axi_arprot=self.axi.ar.prot,
            o_m_axi_arqos=self.axi.ar.qos,
            i_m_axi_rvalid=self.axi.r.valid, o_m_axi_rready=self.axi.r.ready,
            i_m_axi_rid=self.axi.r.id, i_m_axi_rdata=self.axi.r.data,
            i_m_axi_rresp=self.axi.r.resp, i_m_axi_rlast=self.axi.r.last,
            **ws_ports)
