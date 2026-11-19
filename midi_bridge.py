"""MIDI bridge: audio or MIDI file -> notes -> guitar string -> talk-box mouth.
 - transcribe(audio, sr): monophonic YIN transcriber.
 - render_talkbox(notes): plucked string through the corrected Kelly-Lochbaum tract.
 - Exposes Note, read_midi, and write_midi by importing them from harmonia_guitar.
"""
import numpy as np
from scipy import signal

# Import the shared structures and MIDI I/O from our standalone module
from harmonia_guitar import Note, write_midi, read_midi

# ---------------------------------------------------------------- transcription
_TRANSCRIBER = None

def register_transcriber(fn):
    """Register a callable audio transcriber, or pass None to restore fallback YIN."""
    if fn is not None and not callable(fn):
        raise TypeError("transcriber must be callable or None")
    global _TRANSCRIBER
    _TRANSCRIBER = fn


def _validated_notes(values):
    """Normalize plugin output and enforce the Note contract at the boundary."""
    try:
        items = list(values)
    except TypeError:
        raise ValueError("transcriber must return an iterable of notes") from None
    notes = []
    for item in items:
        try:
            note = item if isinstance(item, Note) else Note(*item)
            start, duration = float(note.start), float(note.dur)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("transcriber returned a malformed note") from None
        if (not np.isfinite(start) or not np.isfinite(duration) or start < 0 or duration <= 0 or
                not isinstance(note.midi, (int, np.integer)) or not 0 <= note.midi <= 127 or
                not isinstance(note.vel, (int, np.integer)) or not 0 <= note.vel <= 127):
            raise ValueError("transcriber returned invalid note timing, pitch, or velocity")
        notes.append(Note(start, duration, int(note.midi), int(note.vel)))
    return notes

def _yin(frame, sr, fmin=70.0, fmax=1200.0, thr=0.15):
    """YIN fundamental frequency estimator."""
    if (not isinstance(sr, (int, np.integer)) or sr <= 0 or
            not np.isfinite(fmin) or not np.isfinite(fmax) or
            not 0 < fmin < fmax or fmax >= sr / 2 or not np.isfinite(thr) or not 0 < thr < 1):
        raise ValueError("invalid YIN sample rate, frequency bounds, or threshold")
    raw = np.asarray(frame)
    if np.iscomplexobj(raw):
        raise ValueError("YIN frame must be real-valued")
    frame = np.asarray(raw, dtype=np.float64)
    if frame.ndim != 1 or len(frame) < 4 or not np.all(np.isfinite(frame)):
        return 0.0, 1.0
    if np.max(np.abs(frame)) <= 1e-12:
        return 0.0, 1.0
    n = len(frame)
    tmax = min(int(sr / fmin), n - 1)
    tmin = max(2, int(sr / fmax))
    if tmax <= tmin:
        return 0.0, 1.0
    
    fft = np.fft.rfft(frame, 2 * n)
    acf = np.fft.irfft(fft * np.conj(fft))[:tmax + 1]
    
    d = np.zeros(tmax + 1)
    sq = np.cumsum(np.r_[0, frame ** 2])
    for tau in range(1, tmax + 1):
        d[tau] = (sq[n - tau] + (sq[n] - sq[tau])) - 2 * acf[tau]
        
    cm = np.ones(tmax + 1)
    run = np.cumsum(d[1:])
    cm[1:] = d[1:] * np.arange(1, tmax + 1) / np.maximum(run, 1e-12)
    
    for tau in range(tmin, tmax):
        if cm[tau] < thr:
            while tau + 1 < tmax and cm[tau + 1] < cm[tau]:
                tau += 1
            a, b, c = cm[tau - 1], cm[tau], cm[tau + 1]
            den = a - 2 * b + c
            t = tau + (0.5 * (a - c) / den if den != 0 else 0.0)
            return sr / t, float(cm[tau])
    return 0.0, 1.0

def transcribe(audio, sr, hop=0.01, win=0.046, min_note=0.06, gate_db=-45.0):
    """Extract Note objects from a finite mono audio array."""
    if not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError("sample rate must be a positive integer")
    if (not np.all(np.isfinite([hop, win, min_note, gate_db])) or
            hop <= 0 or win <= 0 or min_note < 0):
        raise ValueError("hop and window must be positive; min_note must be non-negative")
    if _TRANSCRIBER is not None:
        x = np.asarray(audio)
        if np.iscomplexobj(x) or x.ndim != 1 or not np.all(np.isfinite(x)):
            raise ValueError("audio must be finite mono input")
        return _validated_notes(_TRANSCRIBER(x, sr))

    raw = np.asarray(audio)
    if np.iscomplexobj(raw):
        raise ValueError("audio must be real-valued")
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("audio must be finite mono input")
    H = int(round(hop * sr))
    W = int(round(win * sr))
    if H < 1 or W < 4:
        raise ValueError("hop and window must span at least one and four samples")
    if len(x) < W:
        return []
    pk = np.max(np.abs(x)) + 1e-12
    midis, rms = [], []
    
    for i in range(0, len(x) - W + 1, H):
        f = x[i:i + W]
        r = np.sqrt(np.mean(f ** 2))
        rms.append(r)
        
        if 20 * np.log10(r / pk + 1e-12) > gate_db:
            f0, cm = _yin(f * np.hanning(W), sr)
        else:
            f0, cm = 0.0, 1.0
            
        midis.append(69 + 12 * np.log2(f0 / 440.0) if f0 > 0 and cm < 0.2 else np.nan)
        
    midis = np.array(midis)
    notes = []
    i = 0
    N = len(midis)
    
    while i < N:
        if np.isnan(midis[i]):
            i += 1
            continue
            
        j = i
        seg = [midis[i]]
        
        while j + 1 < N and not np.isnan(midis[j + 1]) and abs(midis[j + 1] - np.median(seg[-5:])) < 0.7:
            j += 1
            seg.append(midis[j])
            
        dur = (j - i + 1) * hop
        if dur >= min_note:
            lvl = np.mean(rms[i:j + 1]) / pk
            vel = int(np.clip(40 + 87 * lvl ** 0.5, 20, 127))
            notes.append(Note(i * hop, dur, int(round(float(np.median(seg)))), vel))
        i = j + 1
        
    return notes

# ---------------------------------------------------------------- guitar + talk box
def render_talkbox(notes, fs=48000, octave_shift=0, tail=0.8, glide=0.04):
    """Pluck notes and pass them through the Talk-Box mouth tract."""
    import harmonia_guitar as G
    import mouth_tract as M

    try:
        notes = list(notes)
    except TypeError:
        raise ValueError("notes must be an iterable of Note objects") from None
    if not notes:
        return np.zeros(0, dtype=np.float64)
    if (not isinstance(fs, (int, np.integer)) or fs <= 0 or
            not isinstance(octave_shift, (int, np.integer)) or
            not np.all(np.isfinite([tail, glide])) or tail < 0 or glide <= 0):
        raise ValueError("invalid render sample rate, octave shift, tail, or glide")
    for note in notes:
        if (not np.all(np.isfinite([note.start, note.dur])) or
                note.start < 0 or note.dur < 0 or
                not isinstance(note.midi, (int, np.integer)) or not 0 <= note.midi <= 127 or
                not isinstance(note.vel, (int, np.integer)) or not 0 <= note.vel <= 127 or
                not 0 <= note.midi + octave_shift <= 127):
            raise ValueError("invalid note timing, pitch, or velocity")

    n = int(fs * (max(x.start + x.dur for x in notes) + tail))
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    src = np.zeros(n)
    curve = np.zeros(n)
    
    for i, nt in enumerate(sorted(notes, key=lambda z: z.start)):
        a = int(nt.start * fs)
        # G.pluck now maps directly to our updated Karplus-Strong
        velocity = 0.0 if nt.vel == 0 else 0.25 + 0.7 * nt.vel / 127
        p = G.pluck(nt.midi + octave_shift, nt.dur + tail, velocity, fs, seed=1000 + i)
        src[a:a + len(p)] += p[:n - a]
        
        # Vowel curve: pitch class determines the ah..oo axis
        curve[a:] = (nt.midi % 12) / 12.0
        
    # Apply a low-pass filter to smooth the curve transitions
    curve = signal.lfilter([1 - np.exp(-1 / (glide * fs))], [1, -np.exp(-1 / (glide * fs))], curve)
    
    y = M.Tract().process(src, curve)
    
    # Normalize
    return y / (np.max(np.abs(y)) + 1e-12) * 0.7
