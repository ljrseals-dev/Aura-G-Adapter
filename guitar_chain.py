"""Full guitar path for AURA-V:
audio -> AURA-G transcription -> tuned waveguide string -> talk-box tract -> LA-2A leveler -> loudness match -> true-peak limiter -> ProDither 16-bit.
Offline. Output rate 44.1 kHz (the rate the ProDither noise-shaping coefficients are verified at)."""
import numpy as np
from scipy import signal
import aura_g_adapter as A, jowiphon_noise as JN, the_order_aura as O
from la2a_py import LA2ACore

def match_loudness(y, sr, lim, target, tol=0.1, iters=10):
    """Raise the pre-limiter gain until the LIMITED signal measures the target (a plain trim cannot reach it for plucked material)."""
    pre = 10 ** ((target - JN.integrated_lufs(y, sr)) / 20)
    for _ in range(iters):
        z = lim.process_block(y * pre); L = JN.integrated_lufs(z, sr)
        if abs(L - target) <= tol: break
        pre *= 10 ** ((target - L) / 20)
    return z, 20 * np.log10(np.max(np.abs(y * pre)) / np.max(np.abs(z)))      # limited signal, peak gain reduction in dB

def guitar_chain(audio, sr=48000, target_lufs=-14.0, ceiling_dbtp=-1.0, la2a_pr=0.30, seed=0, out_sr=44100):
    notes = A.aura_g_transcribe(audio, sr)
    y = A.render_talkbox_aura(notes, 48000, seed=seed)
    assert out_sr == 44100, "ProDither's noise-shaping coefficients are verified at 44.1 kHz only"
    y = signal.resample_poly(y, 147, 160)                                   # 48 -> 44.1 kHz
    y = LA2ACore(out_sr, 0.0, la2a_pr).process(y)
    lim = O.TruePeakLimiterBlock(out_sr, ceiling_dbtp=ceiling_dbtp)
    yl, gr = match_loudness(y, out_sr, lim, target_lufs)
    out = O.ProDitherBlock(out_sr, seed=seed).process_block(yl)
    return dict(notes=notes, limited=yl, out=out, limiter_gr_db=gr, limiter=lim)
