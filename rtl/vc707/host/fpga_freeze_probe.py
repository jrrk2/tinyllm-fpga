#!/usr/bin/env python3
"""Exercise the on-chip logic-analyser freeze and read the frozen layer's hout.

Programs a freeze (counter-match on layer/step, OR absolute cycle), pulses
restart, waits for `frozen`, then reads back:
  - status: dbg_cyc (0x071), {frozen, cur_layer} (0x072)
  - the frozen layer's hidden_out via the wr_kind 10/11 BRAM peek
    (0x060 target / 0x062 read), decoded as BFP m * 2^(e-15).

Unlike the sim stream path (sim_shadow / mock weights), the FPGA computes with
the real uploaded DDR3 weights, so these per-layer values are meaningful — diff
against the Python golden (lbfp_freeze_L<NN>_*.hex from sim_rtl_fp_hw.py) to
localise the first divergent layer.

Usage:  python3 host/fpga_freeze_probe.py --layer 3 [--step 0]
        python3 host/fpga_freeze_probe.py --cyc 5000000
"""
import argparse, math, os, socket, struct, sys, time

PEER = ("192.168.1.42", 19783)
REG_SNAP_SEL=0x00A; REG_STEP_SEL=0x064; REG_FREEZE_EN=0x065
REG_TRIG_CYC=0x066; REG_TRIG_CYC_EN=0x067
REG_RESTART=0x1F1; REG_DBG_CYC=0x071; REG_DBG_STAT=0x072
REG_STAGE_SEL=0x070
REG_WEIGHT_HASH=0x04A
REG_BRAM_TARGET=0x060; REG_BRAM_READ=0x062
KIND_HOUT_M=10; KIND_HOUT_E=11; KIND_SNAP_M=12; KIND_SNAP_E=13; KIND_STAGE_E=16
STAGE_NAMES=["n1","q","k","v","attn","o","h1","n2","g","u","mlp","d","hout","hin"]
STAGE_NT  =[60,  60, 20,20, 60,   60,60, 60,  160,160,160, 60, 60,  60]   # NT per stage (smollm360)

_seq=[0]
def nseq():
    _seq[0]=(_seq[0]+1)&0xFF; return _seq[0] or 1
def reg_read(s, addr, retries=4):
    for _ in range(retries):
        sq=nseq()
        s.sendto(struct.pack("<BBHBBBB",0x02,sq,addr&0xFFFF,1,0,0,0),PEER)
        deadline=time.time()+0.5
        while time.time()<deadline:
            try: buf,_a=s.recvfrom(2048)
            except socket.timeout: continue
            if len(buf)<12 or buf[0]!=0x03 or buf[1]!=sq: continue
            return struct.unpack_from("<I",buf,8)[0]
    raise TimeoutError(f"no REG_RSP 0x{addr:03x}")
def reg_write(s, addr, val):
    s.sendto(struct.pack("<BBB",0x01,nseq(),1)+b"\x00"*5+struct.pack("<HIH",addr&0xFFFF,val&0xFFFFFFFF,0),PEER)
def read_bram(s, kind, addr):
    reg_write(s, REG_BRAM_TARGET, ((kind&0x1f)<<18)|(addr&0x3ffff))
    return reg_read(s, REG_BRAM_READ)&0xFFFF
def s16(v): return v-0x10000 if v&0x8000 else v
def s8(v):  return v-0x100   if v&0x80   else v

def load_ints(p):
    if not os.path.exists(p): return []
    with open(p) as f: return [int(x) for x in f.read().split()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--layer",type=int,default=0)
    ap.add_argument("--step", type=int,default=0)
    ap.add_argument("--cyc",  type=int,default=0)
    ap.add_argument("--d",    type=int,default=960)
    args=ap.parse_args(); D=args.d; NT_D=D//16
    s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.settimeout(0.5)

    bv=reg_read(s,0x10F); print(f"BUILD_VERSION=0x{bv:08x}")
    # Program the freeze.
    if args.cyc:
        reg_write(s,REG_TRIG_CYC,args.cyc); reg_write(s,REG_TRIG_CYC_EN,1); reg_write(s,REG_FREEZE_EN,0)
    else:
        reg_write(s,REG_SNAP_SEL,args.layer); reg_write(s,REG_STEP_SEL,args.step)
        reg_write(s,REG_FREEZE_EN,1); reg_write(s,REG_TRIG_CYC_EN,0)
    time.sleep(0.01)
    reg_write(s,REG_RESTART,1)
    # Wait for frozen (0x072 bit5) or timeout.
    t0=time.time(); frozen=0
    while time.time()-t0<30:
        st=reg_read(s,REG_DBG_STAT)
        if (st>>5)&1: frozen=1; break
        time.sleep(0.05)
    st=reg_read(s,REG_DBG_STAT); cyc=reg_read(s,REG_DBG_CYC)
    cur_layer=st&0x1f; frozen=(st>>5)&1
    whash=reg_read(s,REG_WEIGHT_HASH)
    print(f"frozen={frozen}  cur_layer={cur_layer}  dbg_cyc={cyc}  weight_hash=0x{whash:08x}")
    if not frozen:
        print("FAIL: engine did not freeze"); return 1

    # Read the frozen layer's hidden, decode.  Two paths:
    #   live (10/11) = hout_m/hout_e — overwritten every layer/token, races with
    #                  the engine; this is what gave non-deterministic reads.
    #   snap (12/13) = snap_m/snap_e — latched ONCE at (snap_layer_sel,
    #                  snap_step_sel), never overwritten until restart (option b).
    # SRC=live forces the old path; default reads the stable snapshot.
    src = os.environ.get("SRC", "snap")
    KM, KE = (KIND_HOUT_M, KIND_HOUT_E) if src == "live" else (KIND_SNAP_M, KIND_SNAP_E)
    mant=[s16(read_bram(s,KM,i)) for i in range(D)]
    expo=[s8(read_bram(s,KE,i))  for i in range(NT_D)]
    dec=lambda i: mant[i]*math.ldexp(1.0,expo[i//16]-15)
    vals=[dec(i) for i in range(D)]
    mx=max(vals,key=abs)
    import hashlib
    blob=b''.join((m&0xFFFF).to_bytes(2,'little') for m in mant)+bytes((e&0xFF) for e in expo)
    h=hashlib.md5(blob).hexdigest()[:8]
    print(f"hout[{src}]: md5={h}  max|val|={abs(mx):.4g}  max_exp={max(expo)}  [0..7]:", " ".join(f"{dec(i):+.4f}" for i in range(8)))

    # Per-stage exponent sweep — find which stage's per-tile exponent saturates.
    if os.environ.get("STAGES"):
        print("  per-stage max exponent (saturation ~ +/-127 = the bug):")
        for sidx, name in enumerate(STAGE_NAMES):
            reg_write(s, REG_STAGE_SEL, sidx); time.sleep(0.002)
            nt = STAGE_NT[sidx]
            es = [s8(read_bram(s, KIND_STAGE_E, i)) for i in range(nt)]
            print(f"    {sidx:2d} {name:4s}: max_exp={max(es):4d}  min_exp={min(es):4d}")

    gm=load_ints(f"../generated/lbfp_freeze_L{args.layer:02d}_m.hex")
    ge=load_ints(f"../generated/lbfp_freeze_L{args.layer:02d}_e.hex")
    if gm and ge:
        gdec=lambda i: gm[i]*math.ldexp(1.0,ge[i//16]-15)
        print("gold[0..7] decoded:", " ".join(f"{gdec(i):+.4f}" for i in range(8)))
        fails=sum(1 for i in range(D) if abs(dec(i)-gdec(i))>1e-3*(abs(gdec(i))+1e-6))
        worst=max((abs(dec(i)-gdec(i)) for i in range(D)),default=0)
        print(f"vs golden L{args.layer:02d}: {fails}/{D} differ (0.1% tol), worst {worst:.4f}")
    else:
        print(f"(no golden lbfp_freeze_L{args.layer:02d}_* — control-only)")
    return 0

if __name__=="__main__":
    sys.exit(main())
