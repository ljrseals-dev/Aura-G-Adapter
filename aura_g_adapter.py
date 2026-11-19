"""AURA-G adapter: (1) transcription through AURA-G's own front end with onset debouncing and note durations,
(2) a tuned waveguide string that replaces DigitalWaveguideString for synthesis (integer-delay tuning error, pitch-independent-of-fret and
over-strong fret damping are fixed), (3) rendering through the corrected talk-box tract.  Needs only numpy/scipy (no torch)."""
import numpy as np
from scipy import signal
from aura_g.front_end import FrontEndAnalyzer
import aura_g.front_end as _FE
import midi_bridge as B

def fixed_yin(frame, sr, fmin=70.0, fmax=1200.0, threshold=0.15):
    """YIN with the standard rule: take the FIRST lag whose normalised difference is below threshold (then walk down to its minimum).
    AURA-G's yin_pitch takes the lowest minimum among all candidate lags, which lands on 2x/3x the period for periodic plucks (octave-low errors)."""
    x = np.asarray(frame, dtype=np.float64) - np.mean(frame); n = len(x); tau_min = max(2, int(sr / fmax)); tau_max = int(sr / fmin)
    if tau_max >= n - 2: return 0.0, 0.0
    f = np.fft.rfft(x, 2 * n); acf = np.fft.irfft(f * np.conj(f))[:tau_max + 1]; sq = np.concatenate(([0.0], np.cumsum(x * x)))
    tau = np.arange(1, tau_max + 1); diff = np.zeros(tau_max + 1); diff[1:] = sq[n - tau] + (sq[n] - sq[tau]) - 2 * acf[1:tau_max + 1]
    cm = np.ones(tau_max + 1); run = np.cumsum(diff[1:]); cm[1:] = diff[1:] * tau / np.maximum(run, 1e-12)
    for t in range(tau_min, tau_max):
        if cm[t] < threshold:
            while t + 1 < tau_max and cm[t + 1] < cm[t]: t += 1
            a, b, c = cm[t - 1], cm[t], cm[t + 1]; den = a - 2 * b + c; tt = t + (0.5 * (a - c) / den if den != 0 else 0.0)
            return sr / tt, float(1.0 - cm[t])
    return 0.0, 0.0

_FE.yin_pitch = fixed_yin   # patch AURA-G's front end in place (see docstring)

def aura_g_transcribe(audio, sr, hop=480, min_gap=0.10, min_dur=0.06, gate=0.02):
    """AURA-G front end -> notes. The stock TranscriptionEngine emits one event per onset *frame* (about 8 per note) and no durations,
    so notes are rebuilt here from the front end's own pitch and onset tracks."""
    if sr != 48000:
        audio = signal.resample_poly(np.asarray(audio, dtype=np.float64), 48000, sr); sr = 48000
    hist = list(FrontEndAnalyzer(sr=sr, hop=hop).process(np.asarray(audio, dtype=np.float64)))
    dt = hop / sr; notes = []; cur = None
    for i, bus in enumerate(hist):
        pg = bus.get_pitch_gesture(); f0 = 440.0 * 2 ** (pg[0] * 6.0) if pg[0] != 0 else 0.0
        midi = int(round(69 + 12 * np.log2(f0 / 440.0))) if f0 > 0 else None; dyn = pg[6]; onset = pg[1] > 0.5; t = i * dt
        if cur is not None and (midi is None or dyn < gate or midi != cur["midi"] or (onset and t - cur["start"] > min_gap)):
            cur["end"] = t; notes.append(cur); cur = None
        if cur is None and midi is not None and dyn >= gate:
            cur = {"start": t, "midi": midi, "vel": dyn, "end": t}
        elif cur is not None:
            cur["vel"] = max(cur["vel"], dyn)
    if cur is not None: cur["end"] = len(hist) * dt; notes.append(cur)
    return [B.Note(n["start"], n["end"] - n["start"], n["midi"], int(np.clip(30 + 97 * min(1.0, n["vel"]), 20, 127))) for n in notes if n["end"] - n["start"] >= min_dur]

def register(): B.register_transcriber(aura_g_transcribe)

class TunedWaveguideString:
    """Digital waveguide string: ring buffer + 2-tap damping + first-order all-pass fractional delay, tuned to f0 * 2^(fret/12)."""
    def __init__(self, sr=48000, seed=0): self.sr = sr; self.rng = np.random.default_rng(seed)
    def render(self, f0, dur, fret=0, pluck_pos=0.3, dynamics=0.8, t60=None, pick_lowpass=0.5):
        sr = self.sr; f = f0 * 2 ** (fret / 12.0); n = int(sr * dur)
        D = sr / f - 0.5; L = int(np.floor(D - 0.5)); frac = D - L; eta = (1 - frac) / (1 + frac)
        t60 = t60 if t60 is not None else float(np.clip(4.0 * (110.0 / f) ** 0.35, 0.8, 6.0)); g = 10 ** (-3.0 * D / (t60 * sr))
        w = self.rng.uniform(-1, 1, L + 2); k = max(1, int(round(pluck_pos * D))); ex = w.copy(); ex[k:] -= w[:-k]
        ex = signal.lfilter(*signal.butter(1, min(0.45 * sr, 600 + 7400 * pick_lowpass * (0.5 + 0.5 * dynamics)) / (sr / 2)), ex)
        loop = np.convolve([0.5, 0.5], [eta, 1.0]) * g
        den = np.zeros(L + len(loop)); den[:2] = [1.0, eta]; den[L:L + len(loop)] -= loop
        x = np.zeros(n); x[:len(ex)] = ex[:n]; y = signal.lfilter([1.0, eta], den, x)
        tail = min(int(0.02 * sr), n); y[-tail:] *= np.linspace(1, 0, tail)
        return (y / (np.max(np.abs(y)) + 1e-12) * dynamics).astype(np.float64)

def render_talkbox_aura(notes, fs=48000, tail=0.8, glide=0.04, seed=0):
    import mouth_tract as M
    n = int(fs * (max(x.start + x.dur for x in notes) + tail)); src = np.zeros(n); curve = np.zeros(n); S = TunedWaveguideString(fs, seed)
    for nt in sorted(notes, key=lambda z: z.start):
        a = int(nt.start * fs); p = S.render(440 * 2 ** ((nt.midi - 69) / 12), nt.dur + tail, 0, 0.3, 0.25 + 0.7 * nt.vel / 127)
        src[a:a + len(p)] += p[:n - a]; curve[a:] = (nt.midi % 12) / 12.0
    curve = signal.lfilter([1 - np.exp(-1 / (glide * fs))], [1, -np.exp(-1 / (glide * fs))], curve)
    y = M.Tract().process(src, curve); return y / (np.max(np.abs(y)) + 1e-12) * 0.7
