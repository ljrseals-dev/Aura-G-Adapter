"""AURA-G adapter: corrected transcription, tuned string synthesis, and talk-box rendering."""
import numpy as np
from scipy import signal
from aura_g.front_end import FrontEndAnalyzer
import aura_g.front_end as _FE
import midi_bridge as B


def fixed_yin(frame, sr, fmin=70.0, fmax=1200.0, threshold=0.15):
    """YIN: choose the first CMND threshold crossing, then refine its local minimum."""
    if (not isinstance(sr, (int, np.integer)) or sr <= 0 or
            not np.isfinite(fmin) or not np.isfinite(fmax) or
            not 0 < fmin < fmax or fmax >= sr / 2 or not np.isfinite(threshold) or
            not 0 < threshold < 1):
        raise ValueError("invalid YIN sample rate, frequency bounds, or threshold")
    raw = np.asarray(frame)
    if np.iscomplexobj(raw):
        raise ValueError("YIN frame must be real-valued")
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 1 or len(x) < 4 or not np.all(np.isfinite(x)):
        return 0.0, 0.0
    if np.max(np.abs(x)) <= 1e-12:
        return 0.0, 0.0
    x = x - np.mean(x)
    n = len(x)
    tau_min = max(2, int(sr / fmax))
    tau_max = min(int(sr / fmin), n - 2)
    if tau_max <= tau_min:
        return 0.0, 0.0
    spectrum = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:tau_max + 1]
    squared = np.concatenate(([0.0], np.cumsum(x * x)))
    tau = np.arange(1, tau_max + 1)
    diff = np.zeros(tau_max + 1)
    diff[1:] = squared[n - tau] + (squared[n] - squared[tau]) - 2 * acf[1:tau_max + 1]
    cmnd = np.ones(tau_max + 1)
    cmnd[1:] = diff[1:] * tau / np.maximum(np.cumsum(diff[1:]), 1e-12)
    for candidate in range(tau_min, tau_max):
        if cmnd[candidate] < threshold:
            while candidate + 1 < tau_max and cmnd[candidate + 1] < cmnd[candidate]:
                candidate += 1
            a, b, c = cmnd[candidate - 1:candidate + 2]
            den = a - 2 * b + c
            refined = candidate + (0.5 * (a - c) / den if den != 0 else 0.0)
            return (sr / refined, float(np.clip(1.0 - b, 0, 1))) if refined > 0 else (0.0, 0.0)
    return 0.0, 0.0


_FE.yin_pitch = fixed_yin


def aura_g_transcribe(audio, sr, hop=480, min_gap=0.10, min_dur=0.06, gate=0.02):
    """AURA-G front end -> monophonic notes with onset debouncing and durations."""
    raw = np.asarray(audio)
    if np.iscomplexobj(raw):
        raise ValueError("audio must be real-valued")
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("audio must be a mono, one-dimensional array")
    if not np.all(np.isfinite(x)):
        raise ValueError("audio contains NaN or infinite samples")
    if not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError("sample rate must be a positive integer")
    if (not isinstance(hop, (int, np.integer)) or hop <= 0 or
            not np.all(np.isfinite([min_gap, min_dur, gate])) or
            min_gap < 0 or min_dur < 0 or gate < 0):
        raise ValueError("hop must be a positive integer; min_gap, min_dur, and gate must be finite and non-negative")
    if len(x) == 0:
        return []
    if sr != 48000:
        original_sr = int(sr)
        x = signal.resample_poly(x, 48000, original_sr)
        hop = max(1, int(round(hop * 48000 / original_sr)))
        sr = 48000
    history = list(FrontEndAnalyzer(sr=sr, hop=hop).process(x))
    dt = hop / sr
    notes, current = [], None
    for i, bus in enumerate(history):
        gesture = bus.get_pitch_gesture()
        # Zero log-pitch is A4; confidence carries the unvoiced distinction.
        midi_value = 69.0 + 72.0 * float(gesture[0])
        confidence = bus.get_pitch_confidence()
        midi = int(round(midi_value)) if confidence >= 0.5 and np.isfinite(midi_value) and 0 <= midi_value <= 127 else None
        dynamics, onset, t = float(gesture[6]), gesture[1] > 0.5, i * dt
        if current is not None and (midi is None or dynamics < gate or midi != current["midi"] or (onset and t - current["start"] > min_gap)):
            current["end"] = t
            notes.append(current)
            current = None
        if current is None and midi is not None and dynamics >= gate:
            current = {"start": t, "midi": midi, "vel": dynamics, "end": t}
        elif current is not None:
            current["vel"] = max(current["vel"], dynamics)
    if current is not None:
        current["end"] = len(history) * dt
        notes.append(current)
    return [B.Note(n["start"], n["end"] - n["start"], n["midi"],
                   int(np.clip(30 + 97 * min(1.0, n["vel"]), 20, 127)))
            for n in notes if n["end"] - n["start"] >= min_dur]


def register():
    B.register_transcriber(aura_g_transcribe)


class TunedWaveguideString:
    """Fractional-delay waveguide string tuned to the fretted fundamental."""
    def __init__(self, sr=48000, seed=0):
        if not isinstance(sr, (int, np.integer)) or sr <= 0:
            raise ValueError("sample rate must be a positive integer")
        self.sr = int(sr)
        self.rng = np.random.default_rng(seed)

    def render(self, f0, dur, fret=0, pluck_pos=0.3, dynamics=0.8, t60=None, pick_lowpass=0.5):
        if not np.isfinite(f0) or f0 <= 0 or not np.isfinite(dur) or dur < 0:
            raise ValueError("f0 must be positive and duration non-negative")
        if not isinstance(fret, (int, np.integer)) or fret < 0:
            raise ValueError("fret must be a non-negative integer")
        if (not np.all(np.isfinite([pluck_pos, dynamics, pick_lowpass])) or
                not 0 < pluck_pos < 1 or not 0 <= dynamics <= 1 or
                not 0 <= pick_lowpass <= 1):
            raise ValueError("pluck_pos, dynamics, and pick_lowpass must be finite and in range")
        sr = self.sr
        frequency = f0 * 2 ** (fret / 12.0)
        n = int(sr * dur)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        delay = sr / frequency - 0.5
        if delay < 2:
            raise ValueError("requested frequency is too high for this sample rate")
        length = int(np.floor(delay - 0.5))
        frac = delay - length
        eta = (1 - frac) / (1 + frac)
        t60 = float(t60 if t60 is not None else np.clip(4.0 * (110.0 / frequency) ** 0.35, 0.8, 6.0))
        if not np.isfinite(t60) or t60 <= 0:
            raise ValueError("t60 must be positive")
        gain = 10 ** (-3.0 * delay / (t60 * sr))
        noise = self.rng.uniform(-1, 1, length + 2)
        offset = max(1, min(length + 1, int(round(pluck_pos * delay))))
        excitation = noise.copy()
        excitation[offset:] -= noise[:-offset]
        cutoff = min(0.45 * sr, 600 + 7400 * pick_lowpass * (0.5 + 0.5 * dynamics))
        sos = signal.butter(1, cutoff / (sr / 2), output="sos")
        excitation = signal.sosfilt(sos, excitation)
        loop = np.convolve([0.5, 0.5], [eta, 1.0]) * gain
        denominator = np.zeros(length + len(loop))
        denominator[:2] = [1.0, eta]
        denominator[length:length + len(loop)] -= loop
        source = np.zeros(n)
        source[:min(n, len(excitation))] = excitation[:n]
        output = signal.lfilter([1.0, eta], denominator, source)
        tail = min(int(0.02 * sr), n)
        if tail:
            output[-tail:] *= np.linspace(1, 0, tail)
        peak = np.max(np.abs(output)) if len(output) else 0.0
        return (output / max(peak, 1e-12) * dynamics).astype(np.float64)


def render_talkbox_aura(notes, fs=48000, tail=0.8, glide=0.04, seed=0):
    import mouth_tract as M
    try:
        notes = list(notes)
    except TypeError:
        raise ValueError("notes must be an iterable of Note objects") from None
    if not notes:
        return np.zeros(0, dtype=np.float64)
    if (not isinstance(fs, (int, np.integer)) or fs <= 0 or
            not np.all(np.isfinite([tail, glide])) or tail < 0 or glide <= 0):
        raise ValueError("fs must be a positive integer; tail and glide must be finite with tail >= 0 and glide > 0")
    for note in notes:
        if (not np.all(np.isfinite([note.start, note.dur])) or
                note.start < 0 or note.dur < 0 or
                not isinstance(note.midi, (int, np.integer)) or not 0 <= note.midi <= 127 or
                not isinstance(note.vel, (int, np.integer)) or not 0 <= note.vel <= 127):
            raise ValueError("notes must have finite non-negative timing and valid MIDI pitch/velocity")
    end = max(note.start + note.dur for note in notes)
    n = int(fs * (end + tail))
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    source, curve = np.zeros(n), np.zeros(n)
    string = TunedWaveguideString(fs, seed)
    for note in sorted(notes, key=lambda item: item.start):
        start = int(note.start * fs)
        if start >= n:
            continue
        velocity = 0.0 if note.vel == 0 else 0.25 + 0.7 * note.vel / 127
        pluck = string.render(440 * 2 ** ((note.midi - 69) / 12),
                              note.dur + tail, dynamics=velocity)
        count = min(len(pluck), n - start)
        source[start:start + count] += pluck[:count]
        curve[start:] = (note.midi % 12) / 12.0
    coefficient = np.exp(-1 / (glide * fs))
    curve = signal.lfilter([1 - coefficient], [1, -coefficient], curve)
    output = M.Tract().process(source, curve)
    peak = np.max(np.abs(output)) if len(output) else 0.0
    return output / max(peak, 1e-12) * 0.7
