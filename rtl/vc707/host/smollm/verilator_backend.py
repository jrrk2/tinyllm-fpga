"""Persistent Verilator-backed INT8 matvec engine, callable from Python.

Spawns rtl/vc707/sim/obj_dir/Vpersist as a long-running subprocess and talks
to it over stdin/stdout in the protocol defined by tb_persist.cpp.
"""
import os
import struct
import subprocess
import numpy as np

DEFAULT_BIN = os.path.join(os.path.dirname(__file__),
                           "..", "..", "sim", "obj_dir", "Vpersist")
DEFAULT_BIN = os.path.abspath(DEFAULT_BIN)


class VerilatorMatvecBackend:
    """Wraps the Vpersist subprocess.  One backend per Python session."""

    def __init__(self, binary: str = DEFAULT_BIN):
        if not os.path.exists(binary):
            raise FileNotFoundError(
                f"Verilator backend binary not found: {binary}\n"
                f"Build it with: cd rtl/vc707/sim && make persist")
        self.proc = subprocess.Popen(
            [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self._calls = 0
        self._cycles = 0   # not currently reported back; placeholder

    def linear_int8(self, w_int8: np.ndarray, scale_q15: np.ndarray,
                    x_q15: np.ndarray) -> np.ndarray:
        """Compute y_q15 = matvec(w_int8, scale_q15, x_q15) on the SV engine.

        Shapes:
          w_int8    [out, in]  np.int8
          scale_q15 [out]      np.int16
          x_q15     [in]       np.int16
        Returns:
          y_q15     [out]      np.int16
        """
        assert w_int8.dtype == np.int8 and w_int8.ndim == 2
        out_dim, in_dim = w_int8.shape
        assert out_dim % 16 == 0, f"out_dim={out_dim} must be multiple of 16"
        assert scale_q15.shape == (out_dim,) and scale_q15.dtype == np.int16
        assert x_q15.shape == (in_dim,)    and x_q15.dtype     == np.int16

        # Frame
        self.proc.stdin.write(b"LIN0")
        self.proc.stdin.write(struct.pack("<II", in_dim, out_dim))
        self.proc.stdin.write(np.ascontiguousarray(w_int8).tobytes())
        self.proc.stdin.write(np.ascontiguousarray(scale_q15).tobytes())
        self.proc.stdin.write(np.ascontiguousarray(x_q15).tobytes())
        self.proc.stdin.flush()

        # Response: out_dim int16 values
        n = out_dim * 2
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            chunk = self.proc.stdout.read(n - got)
            if not chunk:
                raise RuntimeError("Vpersist closed pipe unexpectedly")
            view[got : got + len(chunk)] = chunk
            got += len(chunk)
        self._calls += 1
        return np.frombuffer(buf, dtype=np.int16).copy()

    def close(self):
        try:
            self.proc.stdin.write(b"QUIT")
            self.proc.stdin.flush()
        except Exception:
            pass
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def __del__(self):
        try: self.close()
        except Exception: pass

    @property
    def call_count(self) -> int:
        return self._calls


def _smoke():
    """Quick self-test: random inputs, compare DUT to numpy reference."""
    rng = np.random.default_rng(1)
    LANES, IN = 16, 64
    w     = rng.integers(-127, 127, size=(LANES, IN), dtype=np.int8)
    scale = rng.integers(-8192, 8192, size=LANES,    dtype=np.int16)
    x     = rng.integers(-32768, 32767, size=IN,     dtype=np.int16)

    # Reference (bit-exact integer mimic)
    acc = np.zeros(LANES, dtype=np.int64)
    for k in range(IN):
        acc += x[k].astype(np.int64) * w[:, k].astype(np.int64)
    ref = np.zeros(LANES, dtype=np.int16)
    for l in range(LANES):
        scaled  = int(acc[l]) * int(scale[l])
        shifted = scaled >> 15
        ref[l]  = max(-32768, min(32767, shifted))

    be = VerilatorMatvecBackend()
    dut = be.linear_int8(w, scale, x)
    be.close()
    diffs = np.array(dut, np.int64) - np.array(ref, np.int64)
    print("DUT:", dut.tolist())
    print("REF:", ref.tolist())
    print("max |diff|:", int(np.abs(diffs).max()))
    if int(np.abs(diffs).max()) == 0:
        print("PASS — Verilator backend is bit-exact")
    else:
        print("FAIL")


if __name__ == "__main__":
    _smoke()
