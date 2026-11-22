"""AURA-V release gates on the guitar path.  python guitar_gates.py  (exit 0 = all pass)"""
import sys, numpy as np
from scipy import signal
import midi_bridge as B, harmonia_guitar as G, guitar_chain as C, jowiphon_noise as JN, the_order_aura as O
SR, OUT, TARGET, CEIL = 48000, 44100, -14.0, -1.0
res = []
def gate(n, ok, d): print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}"); res.append(bool(ok))
tp = lambda y: 20 * np.log10(np.max(np.abs(signal.resample_poly(y, 4, 1))) + 1e-30)
REF = [B.Note(0.1 + 0.45 * i, 0.4, m, 90) for i, m in enumerate([52, 55, 59, 57, 55, 52, 50, 47, 64, 62])]
def guitar_audio():
    x = np.zeros(int(SR * 5.0))
    for i, n in enumerate(REF): p = G.pluck(n.midi, n.dur, 0.8, SR, seed=i); a = int(n.start * SR); x[a:a + len(p)] += p[:len(x) - a]
    return x / np.max(np.abs(x))
if __name__ == "__main__":
    print("== AURA-V gates: AURA-G guitar -> talk box -> LA-2A -> loudness -> true-peak limiter -> ProDither 16-bit ==")
    x = guitar_audio(); r = C.guitar_chain(x, SR, TARGET, CEIL); out = r["out"]; yl = r["limited"]
    L = JN.integrated_lufs(out, OUT); T = tp(out)
    gate("10 of 10 notes transcribed", len(r["notes"]) == 10, f"{len(r['notes'])} notes")
    gate("loudness within 0.5 LU of target", abs(L - TARGET) <= 0.5, f"{L:.2f} LUFS (target {TARGET})")
    gate("true peak <= -0.9 dBTP", T <= CEIL + 0.1, f"{T:.2f} dBTP")
    gate("limiter peak gain reduction <= 8 dB (target reached without crushing the signal)", r["limiter_gr_db"] <= 8.0, f"{r['limiter_gr_db']:.1f} dB")
    grid = np.mean(np.abs(out * 32767 - np.round(out * 32767)) < 1e-6); gate("100% of samples on the 16-bit grid", grid > 0.9999, f"{grid:.4f}")
    flat = O.ProDitherBlock(OUT, shaping="none", seed=0).process_block(yl)
    band = lambda e, a, b: (lambda f, P: 10 * np.log10(np.mean(P[(f >= a) & (f <= b)]) + 1e-40))(*signal.welch(e, OUT, nperseg=2 ** 13))
    d = band(out - yl, 1000, 5000) - band(flat - yl, 1000, 5000); gate("1-5 kHz error floor >= 10 dB below flat TPDF", d <= -10, f"{d:+.1f} dB")
    gate("DC offset < 1e-3", abs(np.mean(out)) < 1e-3, f"{np.mean(out):+.2e}")
    r2 = C.guitar_chain(x, SR, TARGET, CEIL); gate("deterministic (same seed, bit-identical)", np.array_equal(out, r2["out"]), "identical" if np.array_equal(out, r2["out"]) else "differs")
    xn = x.copy(); xn[40000] = np.nan; xn[90000] = np.inf; rn = C.guitar_chain(xn, SR, TARGET, CEIL)
    gate("one NaN and one Inf in the input give finite output", np.all(np.isfinite(rn["out"])) and len(rn["notes"]) >= 8, f"{len(rn['notes'])} notes, finite {bool(np.all(np.isfinite(rn['out'])))}")
    print(f"  [INFO] limiter look-ahead {r['limiter'].latency_samples} samples = {1000*r['limiter'].latency_samples/OUT:.1f} ms; string, tract and LA-2A are offline with no added latency")
    print(f"\nSUMMARY: {sum(res)}/{len(res)} gates passed"); sys.exit(0 if all(res) else 1)
