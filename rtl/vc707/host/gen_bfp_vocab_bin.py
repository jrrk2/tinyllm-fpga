#!/usr/bin/env python3
"""Bake the SmolLM2 vocab.json into a flat binary the C++ client reads.

The vocab strings in vocab.json use GPT-2-style byte-level BPE encoding
(printable-character map of raw bytes).  We invert that map here so
each entry in the binary holds the raw UTF-8 bytes the token decodes to
— the C++ side can just memcpy and print without re-implementing the
byte<->unicode dance.

Binary format (little-endian):
    char[4]   magic = "BFPV"
    uint32    version = 1
    uint32    n_tokens
    uint32    blob_size
    struct { uint32 offset; uint32 length; }  entries[n_tokens]
    uint8     blob[blob_size]
"""
import argparse, json, struct, sys
from pathlib import Path


def gpt2_bytes_to_unicode():
    """Same map HuggingFace tokenizers / GPT-2 use."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n  = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vocab_json", type=Path,
                    help="Path to vocab.json (HF SmolLM2 snapshot)")
    ap.add_argument("--out", type=Path, default=Path("bfp_vocab.bin"))
    args = ap.parse_args()

    raw = json.loads(args.vocab_json.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        sys.exit("vocab.json must be a {token: id} dict")

    n_tokens = max(raw.values()) + 1
    inv_byte = {u: b for b, u in gpt2_bytes_to_unicode().items()}

    blob = bytearray()
    entries = [None] * n_tokens
    for token_str, tok_id in raw.items():
        try:
            data = bytes(inv_byte[ch] for ch in token_str)
        except KeyError as e:
            sys.exit(f"unmapped char in token id={tok_id}: {e!r}")
        offset = len(blob)
        entries[tok_id] = (offset, len(data))
        blob.extend(data)

    # Empty entries shouldn't happen with a packed vocab, but tolerate gaps.
    for i, e in enumerate(entries):
        if e is None:
            entries[i] = (0, 0)

    with args.out.open("wb") as f:
        f.write(b"BFPV")
        f.write(struct.pack("<III", 1, n_tokens, len(blob)))
        for off, ln in entries:
            f.write(struct.pack("<II", off, ln))
        f.write(bytes(blob))

    print(f"wrote {args.out}: n_tokens={n_tokens}, blob={len(blob)} bytes, "
          f"total={args.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
