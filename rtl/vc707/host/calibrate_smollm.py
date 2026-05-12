#!/usr/bin/env python3
"""Calibration pass for SmolLM2-135M Q1.15 fixed-point inference.

Runs FP32 inference on a calibration prompt, captures the max-abs value
on every per-layer bus the FPGA implements, and prints them so we can
decide between full per-tensor scaling (A) and power-of-2 block-FP (C).

Buses captured per layer (0..NL-1):
  norm1     RMSNorm 1 output (input to Q/K/V matvecs)
  q, k, v   matvec outputs (pre-RoPE)
  attn      attention output (after o_proj — input to first residual)
  hidden1   first residual sum (norm2 input)
  norm2     RMSNorm 2 output (input to gate/up matvecs)
  gate, up  matvec outputs
  mlp       SwiGLU output (input to down matvec)
  down      matvec output (input to second residual)
  hidden_out= second residual sum (output of layer)
"""
import argparse, sys, os, math
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def nearest_pow2(x: float) -> int:
    """Smallest power-of-2 ≥ x; returns the exponent (0,1,2,...).
       e.g. x=2.5 → 2 (covers up to 4.0); x=1.0 → 0."""
    if x <= 1.0: return 0
    return int(math.ceil(math.log2(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Once upon a time there was a princess.")
    ap.add_argument("--margin", type=float, default=1.5,
                    help="Safety margin on observed max-abs (to avoid saturation "
                         "on inputs not seen during calibration)")
    args = ap.parse_args()

    print(f"loading {MODEL_NAME} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32).eval()

    cfg = model.config
    NL = cfg.num_hidden_layers

    # ----------------------------------------------------------------------
    # Capture the inter-block bus values via forward hooks on standard
    # SmolLM2/Llama submodules.  We only see what HF's Llama exposes:
    # input_layernorm out, self_attn out (after o_proj),
    # post_attention_layernorm out, mlp out.  q/k/v/gate/up are buried
    # inside attention/mlp — we hook the individual Linears.
    # ----------------------------------------------------------------------
    stats = {f"L{li}": {} for li in range(NL)}
    handles = []

    def hook_factory(layer_idx, key):
        def h(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            v = float(t.abs().max())
            stats[f"L{layer_idx}"][key] = max(stats[f"L{layer_idx}"].get(key, 0.0), v)
        return h

    for li, L in enumerate(model.model.layers):
        handles.append(L.input_layernorm.register_forward_hook(
            hook_factory(li, "norm1")))
        handles.append(L.self_attn.q_proj.register_forward_hook(
            hook_factory(li, "q")))
        handles.append(L.self_attn.k_proj.register_forward_hook(
            hook_factory(li, "k")))
        handles.append(L.self_attn.v_proj.register_forward_hook(
            hook_factory(li, "v")))
        handles.append(L.self_attn.o_proj.register_forward_hook(
            hook_factory(li, "attn")))
        handles.append(L.post_attention_layernorm.register_forward_hook(
            hook_factory(li, "norm2")))
        handles.append(L.mlp.gate_proj.register_forward_hook(
            hook_factory(li, "gate")))
        handles.append(L.mlp.up_proj.register_forward_hook(
            hook_factory(li, "up")))
        handles.append(L.mlp.down_proj.register_forward_hook(
            hook_factory(li, "down")))
        # Hidden1/hidden_out are the layer input/output residuals — capture
        # via a wrapper around the whole layer.
        def make_layer_hook(li):
            def h(_mod, inp, out):
                # input is hidden_in; out is (hidden_out, ...)
                hi = inp[0] if isinstance(inp, tuple) else inp
                ho = out[0] if isinstance(out, tuple) else out
                stats[f"L{li}"]["hidden_in"]  = max(
                    stats[f"L{li}"].get("hidden_in",  0.0), float(hi.abs().max()))
                stats[f"L{li}"]["hidden_out"] = max(
                    stats[f"L{li}"].get("hidden_out", 0.0), float(ho.abs().max()))
            return h
        handles.append(L.register_forward_hook(make_layer_hook(li)))

    ids = tok(args.prompt, return_tensors="pt").input_ids
    print(f"  prompt tokens: {ids.tolist()[0]}", file=sys.stderr)

    with torch.no_grad():
        _ = model(ids)

    for h in handles: h.remove()

    # ----------------------------------------------------------------------
    # Tabulate.  For each bus, compute:
    #   max-abs across all layers
    #   per-layer max-abs
    #   power-of-2 shift needed  (Q1.15 covers ±1; if max-abs is M with
    #     margin, need shift = ceil(log2(M*margin)) so storage = real / 2^shift)
    # ----------------------------------------------------------------------
    buses = ["hidden_in","norm1","q","k","v","attn",
             "hidden_out","norm2","gate","up","down"]
    # Note: hidden1 = hidden_in + attn (intermediate); we approximate it as
    # max(hidden_in, hidden_out) per layer since the model doesn't expose
    # post-residual1 directly.

    print("\n--- Per-bus max|x| across all layers (FP32 reference) ---")
    print(f"{'bus':>10}  {'max-abs':>10}  {'×margin':>10}  shift  scale=2^shift")
    print("-" * 60)
    overall_shifts = {}
    for b in buses:
        amax = max(stats[f"L{li}"].get(b, 0.0) for li in range(NL))
        scaled = amax * args.margin
        shift = nearest_pow2(scaled)
        overall_shifts[b] = shift
        print(f"{b:>10}  {amax:>10.3f}  {scaled:>10.3f}  {shift:>5}  {2**shift:>5}")

    # Per-layer breakdown (compact)
    print("\n--- Per-layer max|x| (×margin), shift in []  ---")
    header = "  L  " + "  ".join(f"{b:>9}" for b in buses)
    print(header)
    for li in range(NL):
        cells = []
        for b in buses:
            v = stats[f"L{li}"].get(b, 0.0) * args.margin
            sh = nearest_pow2(v)
            cells.append(f"{v:>6.2f}[{sh:1d}]")
        print(f"{li:3d}  " + "  ".join(cells))

    # Final: are scales clustered near powers of 2 (favours C) or scattered (needs A)?
    print("\n--- Approach recommendation ---")
    pow2_dist = []
    for b in buses:
        for li in range(NL):
            v = stats[f"L{li}"].get(b, 0.0) * args.margin
            if v <= 1.0: continue
            log2v = math.log2(v)
            dist = abs(log2v - round(log2v))   # 0 = exactly power of 2
            pow2_dist.append(dist)
    if pow2_dist:
        avg_dist = sum(pow2_dist) / len(pow2_dist)
        print(f"  avg distance from nearest log2 (lower = more pow-of-2 like): {avg_dist:.3f}")
        if avg_dist < 0.20:
            print("  → scales cluster near powers of 2 — Approach C (block FP) is "
                  "almost as accurate as A and much simpler.")
        else:
            print("  → scales are scattered — Approach A (general per-tensor scales) "
                  "gives noticeably better fidelity.")
    print(f"  worst-case shift = {max(overall_shifts.values())}  "
          f"({2**max(overall_shifts.values())}× headroom needed)")


if __name__ == "__main__":
    main()
