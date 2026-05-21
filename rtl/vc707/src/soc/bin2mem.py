#!/usr/bin/env python3
"""bin2mem.py — convert a firmware .bin to a Verilog $readmemh / updatemem .mem
(one 32-bit little-endian word per line, hex).  Used to (a) init the progmem
BRAM at synth and (b) feed `updatemem -data` to swap firmware in a built .bit
without re-synthesis.

Usage:  python3 bin2mem.py firmware.bin > firmware.mem
"""
import struct, sys

data = open(sys.argv[1], "rb").read()
data += b"\x00" * ((-len(data)) % 4)          # pad to a 32-bit boundary
for i in range(0, len(data), 4):
    sys.stdout.write("%08x\n" % struct.unpack("<I", data[i:i+4])[0])
