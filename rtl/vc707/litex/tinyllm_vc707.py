#!/usr/bin/env python3
"""tinyllm on the proven LiteX VC707 SoC.

Base SoC = xc7-bitstream-tools' examples/vc707-litex/vc707_litex.py in the
configuration its Makefile calls vc707-litex-ddr-ethmin-smpsd-1gb (VexRiscv
SMP, 1 GiB DDR3 SODIMM through the V7 PHY, LiteEth over the ethmin SGMII PHY,
LiteSDCard), proven on this board.  This target subclasses it and adds the
tinyllm engine as a peripheral (litex/tinyllm.py): CSRs on the sys bus, a
512-bit AXI read master on its own LiteDRAM port, and a 40 MHz engine clock.

Bare-metal: the firmware (fw/) runs on the VexRiscv; no Linux.

    make litex-engine            # generate + Vivado build
    make litex-engine-gen        # generate only (elaboration check)
"""

import os
import re
import shutil
import sys

LITEX_VC707_DIR = os.environ.get("LITEX_VC707_DIR",
                                 os.path.expanduser("~/xc7-bitstream-tools/examples/vc707-litex"))
sys.path.insert(0, LITEX_VC707_DIR)

from migen import *
from litex.soc.integration.builder import Builder
from litex.soc.interconnect.csr import CSRStatus
from litedram.frontend.axi import LiteDRAMAXI2Native
from litex_boards.platforms import xilinx_vc707

import vc707_litex                       # the proven base target
from vc707_litex import BaseSoC, mac_for_cpu, DDR_SYS_CLK_FREQ
from tinyllm import TinyLLM, read_cfg

HERE      = os.path.dirname(os.path.abspath(__file__))
VC707_DIR = os.path.normpath(os.path.join(HERE, ".."))             # rtl/vc707
RTL_DIR   = os.path.normpath(os.path.join(VC707_DIR, ".."))        # rtl
SMOLLM_DIR    = os.path.join(VC707_DIR, "src", "smollm")
GENERATED_DIR = os.path.join(RTL_DIR, "generated")

# ROM / LUT init files the engine $readmemh's (relative "../generated/x.hex"
# from the gateware dir, or by PREFIX-name through the include path).
ENGINE_HEX = ["exp_lut.hex", "silu_lut.hex"] + [
    f"lbfp_full_{n}.hex" for n in ("PROMPT", "G1_m", "G1_e", "G2_m", "G2_e",
                                   "K_INIT_m", "K_INIT_e", "V_INIT_m", "V_INIT_e",
                                   "NORM_W_m", "NORM_W_e")]

def read_build_version(path):
    with open(path) as f:
        m = re.search(r"BUILD_VERSION\s*=\s*32'h([0-9a-fA-F]+)", f.read())
    return int(m.group(1), 16) if m else 0

ENGINE_CORE_FREQ = 40e6
TINYLLM_MAC = 0x10e2d5544c4d

class TinyLLMSoC(BaseSoC):
    def __init__(self, build_generated_dir, **kwargs):
        BaseSoC.__init__(self, **kwargs)

        # 40 MHz engine clock from the DDR CRG's MMCM (S7MMCM takes extra
        # outputs until it is finalised), reset from the same lock.
        self.crg.cd_tinyllm_core = ClockDomain()
        self.crg.pll.create_clkout(self.crg.cd_tinyllm_core, ENGINE_CORE_FREQ)
        self.platform.add_false_path_constraints(self.crg.cd_sys.clk,
                                                 self.crg.cd_tinyllm_core.clk)

        cfg = read_cfg(os.path.join(GENERATED_DIR, "lbfp_full_cfg.svh"))
        self.tinyllm = TinyLLM(self.platform, cfg,
            smollm_dir    = SMOLLM_DIR,
            generated_dir = build_generated_dir,
            build_version = read_build_version(os.path.join(GENERATED_DIR, "build_version.svh")),
            core_cd="tinyllm_core", axi_cd="sys")

        # The engine's AXI read master on its own LiteDRAM port.  Engine
        # addresses are byte offsets into the SODIMM, i.e. main_ram-relative:
        # the weight image goes at main_ram+0 exactly as the DDR3 image was
        # laid out for the MIG build.
        port = self.sdram.crossbar.get_port(mode="read", data_width=512)
        self.tinyllm_axi2native = LiteDRAMAXI2Native(self.tinyllm.axi, port, base_address=0)
        self.add_constant("TINYLLM_WEIGHTS_OFFSET", 0)
        self.add_constant("TINYLLM_CORE_FREQ", int(ENGINE_CORE_FREQ))
        for k, v in cfg.items():
            self.add_constant(f"TINYLLM_{k}", v)
        for r, b in self.tinyllm.bases.items():
            self.add_constant(f"TINYLLM_BASE_{r.upper()}", b)

def main():
    from litex.build.parser import LiteXArgumentParser
    parser = LiteXArgumentParser(platform=xilinx_vc707.Platform,
                                 description="tinyllm on the LiteX VC707 SoC (bare-metal).")
    parser.add_target_argument("--local-ip",   default=vc707_litex.LOCAL_IP)
    parser.add_target_argument("--remote-ip",  default=vc707_litex.REMOTE_IP)
    parser.add_target_argument("--mac-address", default=None, type=lambda v: int(v, 0))
    parser.add_target_argument("--tftp-port",  default=vc707_litex.TFTP_PORT, type=int)
    parser.add_target_argument("--flow",       default="vivado")
    parser.add_target_argument("--no-sdcard",  action="store_true", help="omit LiteSDCard")
    parser.add_target_argument("--no-led-chaser", action="store_true")
    parser.set_defaults(cpu_type="vexriscv_smp", cpu_variant="linux", cpu_count=1,
                        hardware_breakpoints=0, with_wishbone_memory=True,
                        l2_size=131072, csr_csv="csr.csv", csr_json="csr.json")
    args = parser.parse_args()

    # Own MAC ("TLM"), so the BIOS netboots from its own per-MAC TFTP
    # directory (~/tftp-vc707/10:e2:d5:54:4c:4d) and never collides with the
    # Linux payload of the 1GB SoC that shares this CPU configuration.
    mac_address = args.mac_address
    if mac_address is None:
        mac_address = TINYLLM_MAC
    print("MAC address: %012x" % mac_address)

    builder_kw = parser.builder_argdict
    builder_out = builder_kw.get("output_dir") or os.path.join("build", "xilinx_vc707")
    build_generated_dir = os.path.abspath(os.path.join(builder_out, "generated"))
    os.makedirs(build_generated_dir, exist_ok=True)
    for h in ENGINE_HEX:
        shutil.copy(os.path.join(GENERATED_DIR, h), build_generated_dir)

    soc = TinyLLMSoC(build_generated_dir,
        sys_clk_freq     = DDR_SYS_CLK_FREQ,
        with_led_chaser  = not args.no_led_chaser,
        with_ddr         = True,
        with_ddr64       = True,
        with_ethmin_phy  = True,
        with_sdcard      = not args.no_sdcard,
        local_ip         = args.local_ip,
        remote_ip        = args.remote_ip,
        mac_address      = mac_address,
        tftp_port        = args.tftp_port,
        flow             = args.flow,
        **parser.soc_argdict)
    builder = Builder(soc, **builder_kw)
    if args.build:
        builder.build(**parser.toolchain_argdict)

if __name__ == "__main__":
    main()
