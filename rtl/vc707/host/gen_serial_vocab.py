#!/usr/bin/env python3
"""Build the firmware-side vocabulary table (serial prompt mode) from bfp_vocab.bin.

Layout (little-endian, consumed by fw_engine.c run_prompt/tokenize):
  u32[0] 'SVOC'  [1] n_tokens  [2] order_off  [3] entries_off  [4] blob_off  [5..7] 0
  byte 32:      first-byte index, 256 x {u32 start, u32 count} into `order`
  order_off:    u16 token ids grouped by first byte, longest token first
  entries_off:  u32 per token id = (len << 24) | blob_off   (len <= 255, blob < 16 MB)
  blob_off:     raw token bytes (the bfp_vocab.bin blob, offsets unchanged)

Usage: gen_serial_vocab.py bfp_vocab.bin serial_vocab.bin
Upload: bfp_client -a 0x3C000000 upload serial_vocab.bin   (SVOC_BASE in fw_engine.c)
"""
import struct, sys

def main():
    src, dst = sys.argv[1], sys.argv[2]
    f = open(src, 'rb').read()
    assert f[:4] == b'BFPV', 'bad magic'
    n, blob_size = struct.unpack_from('<II', f, 8)
    ents = [struct.unpack_from('<II', f, 16 + 8 * i) for i in range(n)]
    blob = f[16 + 8 * n: 16 + 8 * n + blob_size]
    assert len(blob) == blob_size
    assert max(l for _, l in ents) < 256 and blob_size < (1 << 24)
    order = sorted((t for t in range(n) if ents[t][1] > 0),
                   key=lambda t: (blob[ents[t][0]], -ents[t][1], t))
    index = []
    i = 0
    for b in range(256):
        start = i
        while i < len(order) and blob[ents[order[i]][0]] == b:
            i += 1
        index.append((start, i - start))
    hdr_len = 32 + 256 * 8
    order_off = hdr_len
    entries_off = order_off + ((2 * len(order) + 3) & ~3)
    blob_off = entries_off + 4 * n
    out = bytearray()
    out += struct.pack('<8I', 0x434F5653, n, order_off, entries_off, blob_off, 0, 0, 0)
    for s, c in index:
        out += struct.pack('<II', s, c)
    out += struct.pack('<%dH' % len(order), *order)
    out += b'\0' * (entries_off - len(out))
    for off, l in ents:
        out += struct.pack('<I', (l << 24) | off)
    out += blob
    out += b'\0' * ((-len(out)) % 64)          # whole 64-byte DDR chunks for the uploader
    open(dst, 'wb').write(out)
    print(f'[serial-vocab] {n} tokens, {len(order)} indexed, {len(out)} bytes -> {dst}')

if __name__ == '__main__':
    main()
