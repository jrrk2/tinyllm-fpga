#!/usr/bin/env python3
"""Assemble the tinyllm SD-card bundle (and the TFTP netboot payload).

    make_sd_bundle.py --generated ../../../generated --vocab ../../release/bfp_vocab.bin \
                      --fw ../fw/tinyllm_fw.bin --out /media/sdcard

Writes into --out:
  WEIGHTS.BIN   copy of lbfp_full_DDR3.bin (the engine's DDR3 image, main_ram+0)
  ROMS.BIN      BRAM ROM set: records {u32 kind, u32 n, u16 data[n], pad to 4}
                kinds 0..6 = G1_m G1_e G2_m G2_e NORM_W_m NORM_W_e PROMPT
                (same order and parsing as bfp_client load-roms)
  VOCAB.BIN     serial-prompt vocab table (host/gen_serial_vocab.py)
  FW.BIN        the firmware, and boot.json so the LiteX BIOS boots it from
                the card ("sdcardboot") -- or, with --tftp, from the network.
"""
import argparse, json, os, shutil, struct, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROMS = [(0, "G1_m"), (1, "G1_e"), (2, "G2_m"), (3, "G2_e"),
        (4, "NORM_W_m"), (5, "NORM_W_e"), (6, "PROMPT")]
FW_ADDR = 0x7F000000          # main_ram + 0x3F000000, see fw/linker.ld

def read_hex(path):
    with open(path) as f:
        return [int(t, 16) & 0xFFFF for t in (l.strip() for l in f) if t]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", default=os.path.join(HERE, "..", "..", "..", "generated"))
    ap.add_argument("--prefix", default="lbfp_full_")
    ap.add_argument("--vocab", default=os.path.join(HERE, "..", "..", "release", "bfp_vocab.bin"),
                    help="bfp_vocab.bin to build VOCAB.BIN from (via host/gen_serial_vocab.py)")
    ap.add_argument("--fw", default=os.path.join(HERE, "..", "fw", "tinyllm_fw.bin"))
    ap.add_argument("--out", required=True, help="SD card mount point (or a staging dir)")
    ap.add_argument("--no-weights", action="store_true", help="skip the 843 MB image copy")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ROMS.BIN
    with open(os.path.join(a.out, "ROMS.BIN"), "wb") as f:
        for kind, name in ROMS:
            data = read_hex(os.path.join(a.generated, f"{a.prefix}{name}.hex"))
            f.write(struct.pack("<II", kind, len(data)))
            f.write(struct.pack(f"<{len(data)}H", *data))
            if len(data) & 1:
                f.write(b"\0\0")
            print(f"ROMS.BIN: kind {kind} {name}: {len(data)} entries")

    # VOCAB.BIN
    gen = os.path.join(HERE, "..", "..", "host", "gen_serial_vocab.py")
    subprocess.check_call([sys.executable, gen, a.vocab, os.path.join(a.out, "VOCAB.BIN")])

    # FW.BIN + boot.json (LiteX BIOS sdcardboot / netboot format)
    shutil.copy(a.fw, os.path.join(a.out, "FW.BIN"))
    with open(os.path.join(a.out, "boot.json"), "w") as f:
        json.dump({"FW.BIN": "0x%08x" % FW_ADDR}, f, indent=1)
        f.write("\n")

    # WEIGHTS.BIN
    if not a.no_weights:
        src = os.path.join(a.generated, f"{a.prefix}DDR3.bin")
        print(f"WEIGHTS.BIN: copying {os.path.getsize(src)} bytes ...")
        shutil.copy(src, os.path.join(a.out, "WEIGHTS.BIN"))
    print("bundle written to", a.out)

if __name__ == "__main__":
    main()
