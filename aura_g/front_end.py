"""Front-End Analyzer — SuperFlux-style onsets + Probabilistic YIN + timbre."""
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter1d
from .latent_bus import LatentBus


def yin_pitch(frame, sr, fmin=70.0, fmax=1200.0, threshold=0.15):
    """YIN fundamental frequency estimator (simplified, probabilistic via CMND)."""
    if (not isinstance(sr, (int, np.integer)) or sr <= 0 or
            not np.all(np.isfinite([fmin, fmax, threshold])) or
            fmin <= 0 or fmax <= fmin or fmax >= sr / 2 or not 0 < threshold < 1):
        raise ValueError("invalid sample rate, pitch bounds, or YIN threshold")
    frame = np.asarray(frame, dtype=np.float64)
    if frame.ndim != 1 or len(frame) < 4 or not np.all(np.isfinite(frame)):
        return 0.0, 0.0
    frame = frame - np.mean(frame)
    tau_min = max(2, int(sr / fmax))
    tau_max = min(int(sr / fmin), len(frame) - 1)
    if tau_max <= tau_min:
        return 0.0, 0.0
    diff = np.zeros(tau_max + 1)
    for tau in range(1, tau_max + 1):
        d = frame[:-tau] - frame[tau:]
        diff[tau] = np.dot(d, d)
    cmnd = np.ones(tau_max + 1)
    running = 0.0
    for tau in range(1, tau_max + 1):
        running += diff[tau]
        cmnd[tau] = diff[tau] * tau / running if running > 0 else 1.0
    candidates = np.flatnonzero(cmnd[tau_min:tau_max + 1] < threshold) + tau_min
    if not len(candidates):
        return 0.0, 0.0
    tau = int(candidates[0])
    while tau + 1 <= tau_max and cmnd[tau + 1] < cmnd[tau]:
        tau += 1
    a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[min(tau + 1, tau_max)]
    den = a - 2 * b + c
    refined = tau + (0.5 * (a - c) / den if den != 0 else 0.0)
    return (sr / refined, float(np.clip(1.0 - b, 0.0, 1.0))) if refined > 0 else (0.0, 0.0)


def spectral_features(frame, sr):
    """Return timbre descriptors normalized to [0,1]."""
    raw = np.asarray(frame)
    if np.iscomplexobj(raw):
        raise ValueError("frame must be real-valued")
    frame = np.asarray(raw, dtype=np.float64)
    if frame.ndim != 1 or len(frame) == 0 or not np.all(np.isfinite(frame)):
        raise ValueError("frame must be a non-empty finite mono array")
    if not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError("sample rate must be a positive integer")
    spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
    freqs = np.fft.rfftfreq(len(frame), 1 / sr)
    total = spec.sum() + 1e-9
    centroid = float((freqs * spec).sum() / total) / (sr / 2)
    cumsum = np.cumsum(spec) / total
    rolloff_idx = np.searchsorted(cumsum, 0.85)
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)]) / (sr / 2)
    flatness = float(np.exp(np.mean(np.log(spec + 1e-9))) / (np.mean(spec) + 1e-9))
    zcr = float(np.mean(np.abs(np.diff(np.sign(frame))) / 2)) if len(frame) > 1 else 0.0
    ac = np.correlate(frame, frame, mode='full')[len(frame) - 1:]
    ac /= (ac[0] + 1e-9)
    harmonic_ratio = float(np.clip(ac[1:].max(), 0, 1)) if len(ac) > 1 else 0.0
    h2 = spec[2] if len(spec) > 2 else 0.0
    h1 = spec[1] if len(spec) > 1 else 1e-9
    even_odd = float(np.clip(h2 / (h1 + 1e-9), 0, 1))
    attack_slope = float(np.clip(np.max(np.abs(frame)) / (np.std(frame) + 1e-9), 0, 1))
    sustain = float(np.clip(np.sqrt(np.mean(frame ** 2)) * 3, 0, 1))
    return dict(centroid=centroid, rolloff=rolloff, flatness=flatness, zcr=zcr,
                harmonic_ratio=harmonic_ratio, even_odd=even_odd,
                attack_slope=attack_slope, sustain=sustain)


class FrontEndAnalyzer:
    def __init__(self, sr=48000, hop=480, win=2048, fmin=70.0, fmax=1200.0):
        if (not isinstance(sr, (int, np.integer)) or sr <= 0 or
                not isinstance(hop, (int, np.integer)) or hop <= 0 or
                not isinstance(win, (int, np.integer)) or win < 4 or
                not np.all(np.isfinite([fmin, fmax])) or fmin <= 0 or fmax <= fmin or fmax >= sr / 2):
            raise ValueError("invalid sample rate, hop, window, or pitch bounds")
        self.sr, self.hop, self.win = int(sr), int(hop), int(win)
        self.fmin, self.fmax = float(fmin), float(fmax)
        self.prev_f0 = 0.0
        self._prev_mag = None

    def _superflux_onset(self, frame):
        mag = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        mag_max = maximum_filter1d(mag, size=3)
        if self._prev_mag is None:
            self._prev_mag = np.zeros_like(mag)
        flux = np.sum(np.maximum(mag_max - self._prev_mag, 0))
        self._prev_mag = mag
        return flux

    def process(self, audio):
        """Yield LatentBus snapshots at hop rate; short input yields no frames."""
        raw = np.asarray(audio)
        if np.iscomplexobj(raw):
            raise ValueError("audio must be real-valued")
        audio = np.asarray(raw, dtype=np.float64)
        if audio.ndim != 1:
            raise ValueError("audio must be a mono, one-dimensional array")
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio contains NaN or infinite samples")
        if len(audio) < self.win:
            return
        n_hops = 1 + (len(audio) - self.win) // self.hop
        onset_env = np.zeros(n_hops)
        f0s = np.zeros(n_hops)
        pitch_confidences = np.zeros(n_hops)
        timbre_feats = []
        for i in range(n_hops):
            start = i * self.hop
            frame = audio[start:start + self.win]
            f0, conf = yin_pitch(frame, self.sr, self.fmin, self.fmax)
            pitch_confidences[i] = conf
            if conf > 0.5 and f0 > 0:
                self.prev_f0 = f0
                f0s[i] = f0
            else:
                f0s[i] = self.prev_f0
            onset_env[i] = self._superflux_onset(frame)
            timbre_feats.append(spectral_features(frame, self.sr))
        onset_thresh = np.mean(onset_env) + 0.8 * np.std(onset_env)
        onsets = (onset_env > onset_thresh).astype(np.float32)
        for i in range(n_hops):
            bus = LatentBus(sample_rate=self.sr, hop=self.hop)
            f0 = f0s[i]
            f0_log = np.log2(f0 / 440.0) / 6.0 if f0 > 0 else 0.0
            bend = 0.0
            if i > 0 and f0s[i - 1] > 0 and f0 > 0:
                bend = np.clip(np.log2(f0 / f0s[i - 1]) * 12, -2, 2) / 2.0
            window = f0s[max(0, i - 25):i + 1]
            vib_depth = float(np.std(window)) / (np.mean(window) + 1e-9)
            dynamics = float(np.clip(np.sqrt(np.mean(audio[i * self.hop:i * self.hop + self.win] ** 2)) * 3, 0, 1))
            pluck_pos = 0.5
            palm_mute = float(1.0 - min(1.0, timbre_feats[i]['centroid'] * 3))
            bus.write_pitch_gesture(f0_log, float(onsets[i]), float(bend), 0.5,
                                    vib_depth, pluck_pos, dynamics, palm_mute)
            bus.write_pitch_confidence(float(pitch_confidences[i]))
            t = timbre_feats[i]
            bus.write_timbre(t['centroid'], t['rolloff'], t['flatness'], t['zcr'],
                             t['harmonic_ratio'], t['even_odd'], t['attack_slope'], t['sustain'])
            yield bus
