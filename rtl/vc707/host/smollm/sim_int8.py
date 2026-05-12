#!/usr/bin/env python3
"""SmolLM2-135M inference with the FPGA's INT8/Q1.15 quantization scheme.

Goal: prove the quantization the RTL implements produces coherent text on
the real model, BEFORE we spend effort wiring up a full SV integration.

Quantization scheme (matches matvec_int8_engine + rmsnorm + softmax_q15):
  - Linear layer weights: per-row symmetric INT8 with FP scale (round-to-nearest)
  - Activations everywhere: Q1.15 fixed point (16-bit, range [-1, 1))
  - All linear ops compute INT32 acc * scale -> Q1.15 output (saturated)
  - RMSNorm, RoPE, attention, softmax, SwiGLU operate in Q1.15 throughout

Run:
    python3 sim_int8.py --prompt "Once upon a time" --tokens 64

A side-by-side FP32 baseline is printed for comparison.
"""
import argparse
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def quantize_q15(x: torch.Tensor) -> torch.Tensor:
    """Round-trip x through Q1.15 (16-bit signed fixed-point, range [-1, 1))."""
    q = (x.clamp(-1.0, 1 - 1/32768) * 32768).round().clamp(-32768, 32767)
    return q / 32768


def quantize_fp16bit_perTensor(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Quantize x to 16-bit signed fixed-point with `scale` = max representable.
    Range is [-scale, scale - scale/32768]."""
    q = (x.clamp(-scale, scale - scale/32768) * (32768.0 / scale)).round().clamp(-32768, 32767)
    return q * (scale / 32768.0)


def quantize_int8_per_row(w: torch.Tensor):
    """Per-row symmetric INT8 quantization.
    Returns (w_int8 as float, scale_per_row)."""
    # w shape [out, in]; quantize each output row independently.
    amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / 127.0
    q = (w / scale).round().clamp(-128, 127)
    return q, scale  # both float; q has integer values stored in float


# ---------------------------------------------------------------------------
# Quantized layer wrappers
# ---------------------------------------------------------------------------

class QLinearINT8(nn.Module):
    """Replaces nn.Linear with INT8 weight × Q1.15 activation matmul.

    Computes y = quantize_q15( (x_q15 @ w_int8.T * scale) ).
    Bias (if any) is quantized to FP and added before the final Q1.15 cast.
    """
    def __init__(self, base: nn.Linear):
        super().__init__()
        with torch.no_grad():
            w = base.weight.detach().clone()
            q_int8, scale = quantize_int8_per_row(w)
            # Store the quantized weight as int8 (multiplied back by scale at use).
            self.register_buffer("w_q",   q_int8.to(torch.int8))
            self.register_buffer("scale", scale.to(torch.float32))
            if base.bias is not None:
                self.register_buffer("bias", base.bias.detach().clone().float())
            else:
                self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct float weight from int8 + per-row scale.
        w_float = self.w_q.to(torch.float32) * self.scale  # [out, in]
        y = x.float() @ w_float.T  # [..., out]
        if self.bias is not None:
            y = y + self.bias
        return y


def patch_to_int8(model: nn.Module):
    """Replace every nn.Linear with QLinearINT8 in-place."""
    n = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, child_name, QLinearINT8(child))
                n += 1
    print(f"  patched {n} Linear layers to INT8", file=sys.stderr)


# ---------------------------------------------------------------------------
# Verilator-backed Linear: every forward sends the activation through
# the SV matvec_int8_engine running in a persistent subprocess.
# ---------------------------------------------------------------------------

class VerilatorLinearINT8(nn.Module):
    def __init__(self, base: nn.Linear, backend, x_scale: float, y_scale: float):
        super().__init__()
        with torch.no_grad():
            w = base.weight.detach().cpu().numpy().astype(np.float32)  # [out, in]
            amax = np.maximum(np.abs(w).max(axis=1), 1e-8)
            ws_real = amax / 127.0                                     # per-row weight scale
            w_int8 = np.round(w / ws_real[:, None]).clip(-128, 127).astype(np.int8)
            # Engine: y_q15 = ((x_q15 @ w_int8.T) * scale_q15) >> 15
            # Real:   y_real = (acc) * x_scale * ws_real / 32768
            # Want:   y_q15 = y_real * 32768 / y_scale
            #              = acc * x_scale * ws_real / y_scale
            # So:     scale_q15 = round(x_scale * ws_real * 32768 / y_scale)
            scale_q15 = np.round(x_scale * ws_real * 32768.0 / y_scale)
            scale_q15 = np.clip(scale_q15, -32768, 32767).astype(np.int16)
        self.w_int8    = w_int8
        self.scale_q15 = scale_q15
        self.x_scale   = float(x_scale)
        self.y_scale   = float(y_scale)
        self.bias      = base.bias.detach().clone() if base.bias is not None else None
        self.backend   = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        in_dim  = shape[-1]
        out_dim = self.w_int8.shape[0]
        x_flat = x.reshape(-1, in_dim).detach().cpu().numpy().astype(np.float32)

        # Quantize x to int16 with x_scale full-scale.
        scale_x = 32768.0 / self.x_scale
        x_q15 = np.round(x_flat * scale_x).clip(-32768, 32767).astype(np.int16)

        out = np.empty((x_q15.shape[0], out_dim), dtype=np.int16)
        for i in range(x_q15.shape[0]):
            out[i] = self.backend.linear_int8(self.w_int8, self.scale_q15,
                                              np.ascontiguousarray(x_q15[i]))
        # Dequantize
        y = out.astype(np.float32) * (self.y_scale / 32768.0)
        y_t = torch.from_numpy(y).reshape(*shape[:-1], out_dim).to(x.device).to(x.dtype)
        if self.bias is not None:
            y_t = y_t + self.bias
        return y_t


def calibrate_linear_scales(model: nn.Module, prompt_ids: torch.Tensor,
                             margin: float = 1.5):
    """Run a forward pass and capture max|x| / max|y| for every nn.Linear."""
    stats = {}      # name -> dict(x=float, y=float)
    handles = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            stats[name] = {"x": 0.0, "y": 0.0}
            def make(name):
                def hook(_mod, inp, out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    stats[name]["x"] = max(stats[name]["x"], float(x.abs().max()))
                    stats[name]["y"] = max(stats[name]["y"], float(out.abs().max()))
                return hook
            handles.append(m.register_forward_hook(make(name)))
    with torch.no_grad():
        _ = model(prompt_ids)
    for h in handles: h.remove()
    # Apply margin so calibration prompt itself doesn't saturate.
    for v in stats.values():
        v["x"] = max(v["x"] * margin, 1e-3)
        v["y"] = max(v["y"] * margin, 1e-3)
    return stats


def patch_to_verilator(model: nn.Module, backend, scales: dict):
    """Replace every nn.Linear with a Verilator-backed wrapper."""
    n = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                full = f"{name}.{child_name}" if name else child_name
                sc = scales.get(full, {"x": 8.0, "y": 8.0})
                setattr(mod, child_name,
                        VerilatorLinearINT8(child, backend, sc["x"], sc["y"]))
                n += 1
    print(f"  patched {n} Linear layers to Verilator backend", file=sys.stderr)


# ---------------------------------------------------------------------------
# Q1.15 activation hooks (post-RMSNorm, post-RoPE, post-softmax, post-SwiGLU)
# ---------------------------------------------------------------------------

def add_calibration_hooks(model: nn.Module):
    """Attach hooks that record max-abs of every RMSNorm/Attn/MLP output."""
    stats = {}  # name -> running max
    def make_hook(name):
        def hook(_mod, _inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            v = float(t.abs().max())
            stats[name] = max(stats.get(name, 0.0), v)
            return out
        return hook
    for name, mod in model.named_modules():
        cn = mod.__class__.__name__
        if "RMSNorm" in cn or cn in ("LlamaMLP", "LlamaAttention", "LlamaSdpaAttention"):
            mod.register_forward_hook(make_hook(name))
    return stats


def add_act_hooks(model: nn.Module, fmt_scale: float):
    """Quantize each inter-block bus to 16-bit signed fixed-point with a
    fixed (non-calibrated) full-scale value."""
    n_norm = 0; n_act = 0
    def hook(_mod, _inp, out):
        if isinstance(out, torch.Tensor):
            return quantize_fp16bit_perTensor(out.float(), fmt_scale)
        if isinstance(out, tuple):
            return tuple(
                quantize_fp16bit_perTensor(o.float(), fmt_scale)
                if isinstance(o, torch.Tensor) else o for o in out
            )
        return out
    for mod in model.modules():
        cn = mod.__class__.__name__
        if "RMSNorm" in cn:
            mod.register_forward_hook(hook); n_norm += 1
        elif cn in ("LlamaMLP", "LlamaAttention", "LlamaSdpaAttention"):
            mod.register_forward_hook(hook); n_act += 1
    print(f"  attached {n_norm} RMSNorm + {n_act} attn/MLP hooks at "
          f"+/-{fmt_scale} 16-bit fixed-point", file=sys.stderr)


def add_calibrated_act_hooks(model: nn.Module, scales_by_name: dict, margin: float = 1.5):
    """Quantize each inter-block bus with its OWN calibrated scale."""
    n = 0
    for name, mod in model.named_modules():
        cn = mod.__class__.__name__
        if "RMSNorm" in cn or cn in ("LlamaMLP", "LlamaAttention", "LlamaSdpaAttention"):
            base = scales_by_name.get(name, 8.0)
            scale = base * margin
            def make_hook(s):
                def h(_mod, _inp, out):
                    if isinstance(out, torch.Tensor):
                        return quantize_fp16bit_perTensor(out.float(), s)
                    if isinstance(out, tuple):
                        return tuple(
                            quantize_fp16bit_perTensor(o.float(), s)
                            if isinstance(o, torch.Tensor) else o for o in out
                        )
                    return out
                return h
            mod.register_forward_hook(make_hook(scale))
            n += 1
    print(f"  attached {n} per-tensor calibrated hooks (margin={margin})",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(model, tok, prompt, max_new):
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new,
                             do_sample=False, num_beams=1,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--skip-fp32", action="store_true")
    ap.add_argument("--mode", choices=("int8w", "q15", "calq15", "verilator", "both"),
                    default="int8w",
                    help="int8w: INT8 weights only.  q15: + uniform 16-bit acts.  "
                         "calq15: + per-tensor calibrated 16-bit acts.  "
                         "verilator: every Linear runs through Vpersist SV engine.")
    ap.add_argument("--cal-margins", type=float, nargs="+", default=[1.0, 1.5, 2.0],
                    help="Margins to apply to calibrated max in calq15 mode")
    ap.add_argument("--act-scales", type=float, nargs="+", default=[1.0, 4.0, 8.0, 16.0],
                    help="Full-scale magnitudes to sweep when --mode includes q15")
    ap.add_argument("--calibrate", action="store_true",
                    help="Run a calibration pass to print per-bus max-abs values")
    args = ap.parse_args()

    print(f"loading {MODEL_NAME} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_fp = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                    torch_dtype=torch.float32)
    model_fp.eval()

    if not args.skip_fp32:
        print("\n=== FP32 baseline ===")
        text = generate(model_fp, tok, args.prompt, args.tokens)
        print(text)

    cal_stats = None
    if args.calibrate or args.mode == "calq15":
        print("\n=== Calibration: per-bus max-abs ===")
        m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        m.eval()
        cal_stats = add_calibration_hooks(m)
        _ = generate(m, tok, args.prompt, 16)
        from collections import defaultdict
        buckets = defaultdict(list)
        for name, v in cal_stats.items():
            suffix = name.rsplit(".", 1)[-1] if "." in name else name
            buckets[suffix].append(v)
        for suffix in sorted(buckets):
            vs = buckets[suffix]
            print(f"  {suffix:30s}  n={len(vs):3d}  max={max(vs):8.3f}  "
                  f"median={sorted(vs)[len(vs)//2]:8.3f}")

    if args.mode in ("int8w", "both"):
        print("\n=== INT8 weights only (FP activations) ===")
        # Reload to get fresh FP weights
        m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        m.eval()
        patch_to_int8(m)
        text = generate(m, tok, args.prompt, args.tokens)
        print(text)

    if args.mode in ("q15", "both"):
        for s in args.act_scales:
            print(f"\n=== INT8 weights + 16-bit activations (full-scale +/-{s}) ===")
            m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
            m.eval()
            patch_to_int8(m)
            add_act_hooks(m, fmt_scale=s)
            text = generate(m, tok, args.prompt, args.tokens)
            print(text)

    if args.mode == "calq15":
        if cal_stats is None:
            print("Need --calibrate for calq15 mode", file=sys.stderr); return
        for margin in args.cal_margins:
            print(f"\n=== INT8 weights + per-tensor calibrated 16-bit (margin={margin}) ===")
            m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
            m.eval()
            patch_to_int8(m)
            add_calibrated_act_hooks(m, cal_stats, margin=margin)
            text = generate(m, tok, args.prompt, args.tokens)
            print(text)

    if args.mode == "verilator":
        from verilator_backend import VerilatorMatvecBackend
        print("\n=== INT8 + Verilator-backed matvec_int8_engine ===")
        # Calibration pass on FP32 to get per-Linear scales
        m_cal = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        m_cal.eval()
        ids = tok(args.prompt, return_tensors="pt").input_ids
        print("  calibrating per-Linear x/y scales ...", file=sys.stderr)
        scales = calibrate_linear_scales(m_cal, ids, margin=1.5)
        del m_cal
        # Build the live model
        m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        m.eval()
        backend = VerilatorMatvecBackend()
        patch_to_verilator(m, backend, scales)
        t0 = time.time()
        text = generate(m, tok, args.prompt, args.tokens)
        elapsed = time.time() - t0
        print(text)
        print(f"\n  generated {args.tokens} tokens in {elapsed:.1f}s "
              f"({args.tokens / elapsed:.2f} tok/s, "
              f"{backend.call_count} matvec subprocess calls)",
              file=sys.stderr)
        backend.close()


if __name__ == "__main__":
    main()
