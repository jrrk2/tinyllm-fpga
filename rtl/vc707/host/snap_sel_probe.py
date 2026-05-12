#!/usr/bin/env python3
"""Probe 0x00A regmap behavior: write each value 0..29, read back, verify."""
import socket, struct, sys, time

PEER  = ("192.168.1.42", 19783)
REG   = 0x00A

_seq = [10]
def nxt():
    _seq[0] = (_seq[0] + 1) & 0xFF
    return _seq[0] or 1

def reg_read(s, addr):
    sq = nxt()
    body = struct.pack("<BBHBBBB", 0x02, sq, addr & 0xFFFF, 1, 0, 0, 0)
    s.sendto(body, PEER)
    deadline = time.time() + 0.5
    while time.time() < deadline:
        try: buf, _ = s.recvfrom(2048)
        except socket.timeout: continue
        if len(buf) < 8 or buf[0] != 0x03 or buf[1] != sq: continue
        return struct.unpack_from("<I", buf, 8)[0]
    return None

def reg_write(s, addr, value):
    # Per microgpt_eth_ctrl.sv line 18:
    #   FT_REG_WRITE payload[2]=n_writes, [8..]=n*{addr16, data32, pad16}
    # Entries start at byte 8 (5 bytes pad after header), 8 bytes per entry.
    pkt = struct.pack("<BBB", 0x01, nxt(), 1) + b"\x00" * 5 + \
          struct.pack("<HIH", addr & 0xFFFF, value & 0xFFFFFFFF, 0)
    s.sendto(pkt, PEER)

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(1.0)
# Most reliable regmap test: write 0x004 = {temperature[31:16], max_gen[15:8], 0},
# then read 0x00C which exposes host_temperature_reg @ [31:16] and
# host_max_gen_reg @ [15:8] — same eth-clk regs, no CDC, direct readback.
print(f"--- control test: write 0x004, read 0x00C diagnostic mirror ---")
print(f"  initial 0x00C = 0x{reg_read(s, 0x00C):08x}")
for temp, mg in [(0xDEAD, 0x42), (0xCAFE, 0xAB), (0x0080, 0x0F)]:
    v = (temp << 16) | (mg << 8) | 0  # bits [7:0] are config flags, ignore
    reg_write(s, 0x004, v)
    time.sleep(0.01)
    rb = reg_read(s, 0x00C)
    rb_temp = (rb >> 16) & 0xFFFF
    rb_mg   = (rb >> 8)  & 0xFF
    print(f"  wrote temp=0x{temp:04x} mg=0x{mg:02x} -> read temp=0x{rb_temp:04x} mg=0x{rb_mg:02x}  "
          f"({'OK' if rb_temp == temp and rb_mg == mg else 'MISMATCH'})")
print()
print(f"--- target: 0x{REG:03x} (snapshot_layer_sel_reg) ---")
print(f"  initial 0x{REG:03x} = 0x{reg_read(s, REG):08x}")
for v in [0, 5, 17, 29, 30, 31, 0]:
    reg_write(s, REG, v)
    time.sleep(0.01)
    rb = reg_read(s, REG)
    ok = "OK" if rb == v else "MISMATCH"
    print(f"  wrote {v:2d} -> read 0x{rb:08x}  ({ok})")
