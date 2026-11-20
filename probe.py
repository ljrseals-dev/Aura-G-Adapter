"""Builds AuraGEngine with g++ and checks waveguide tuning and decay.  python probe.py   (exit 0 = pass)"""
import ctypes, os, subprocess, sys, tempfile
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); so = os.path.join(tempfile.gettempdir(), "auragengine_probe.so")
subprocess.check_call(["g++", "-O2", "-std=c++17", "-Wall", "-shared", "-fPIC", os.path.join(here, "AuraGEngine.cpp"), os.path.join(here, "wrap.cpp"), "-o", so])
L = ctypes.CDLL(so); L.run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_double] + [ctypes.c_float] * 5; sr = 48000
def run(x, wg=196.0, damp=0.995):
    x = np.ascontiguousarray(x, dtype=np.float32); y = np.zeros_like(x); L.run(x.ctypes.data, y.ctypes.data, len(x), sr, wg, damp, 0.0, 0.0, 1.0); return y.astype(float)
ok = True
for wg in (82.41, 196.0, 330.0, 440.0, 659.26, 880.0):
    xi = np.zeros(sr, dtype=np.float32); xi[:48] = 0.5; yy = run(xi, wg)
    S = np.abs(np.fft.rfft(yy * np.hanning(len(yy)), 1 << 19)); f = np.fft.rfftfreq(1 << 19, 1 / sr); m = (f > wg * .8) & (f < wg * 1.2); i = int(np.argmax(np.where(m, S, 0)))
    a, b, c = np.log(S[i - 1]), np.log(S[i]), np.log(S[i + 1]); fm = f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0]); cents = 1200 * np.log2(fm / wg)
    print(f"set {wg:7.2f} Hz  measured {fm:7.2f}  {cents:+.2f} cents"); ok &= abs(cents) < 1.0
t = np.arange(sr * 3) / sr; x = np.zeros(sr * 3, dtype=np.float32); x[:2400] = 0.5 * np.sin(2 * np.pi * 196 * t[:2400]); y = run(x, damp=0.98)
r = lambda a, b: np.sqrt(np.mean(y[int(a * sr):int(b * sr)] ** 2)) + 1e-18; d = 20 * np.log10(r(.2, .25) / r(1, 1.05)); print(f"decay 0.2 s -> 1.0 s at wgDamp 0.98: {d:.1f} dB"); ok &= d > 15 and bool(np.all(np.isfinite(y)))
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
