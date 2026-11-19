"""Digital waveguide with fret-aware tuning and bounded damping."""
import numpy as np


class DigitalWaveguideString:
    def __init__(self, sr=48000, f0=196.0, n_strings=6, fret_damping_scale=1.0):
        if sr <= 0 or f0 <= 0 or fret_damping_scale < 0:
            raise ValueError("sr and f0 must be positive; damping scale non-negative")
        self.sr, self.f0 = int(sr), float(f0)
        self.n_strings = int(n_strings)
        self.fret_damping_scale = float(fret_damping_scale)
        self.delay = max(2, int(round(self.sr / self.f0)))
        self.buffer = np.random.default_rng(0).standard_normal(self.delay) * 0.01
        self._cursor = 0

    def fret_damping(self, fret):
        if not isinstance(fret, (int, np.integer)) or fret < 0:
            raise ValueError("fret must be a non-negative integer")
        return float(np.clip(1.0 - 0.05 * fret * self.fret_damping_scale, 0.0, 1.0))

    def pluck(self, fret=0, pluck_pos=0.5, dynamics=1.0):
        if not 0 < pluck_pos < 1 or not 0 <= dynamics <= 1:
            raise ValueError("pluck_pos must be in (0, 1) and dynamics in [0, 1]")
        frequency = self.f0 * 2 ** (fret / 12.0)
        self.delay = max(2, int(round(self.sr / frequency)))
        idx = int(np.clip(round(pluck_pos * self.delay), 1, self.delay - 1))
        excitation = np.zeros(self.delay, dtype=np.float64)
        excitation[:idx] = np.linspace(0, 1, idx, endpoint=False)
        excitation[idx:] = np.linspace(1, 0, self.delay - idx)
        self.buffer = excitation * dynamics
        self._cursor = 0

    def step(self, n_samples, fret=0):
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        if len(self.buffer) < 2:
            self.buffer = np.pad(self.buffer, (0, 2 - len(self.buffer)))
        output = np.empty(n_samples, dtype=np.float64)
        damping = self.fret_damping(fret)
        size = len(self.buffer)
        for i in range(n_samples):
            current = self._cursor
            nxt = (current + 1) % size
            output[i] = self.buffer[current]
            self.buffer[current] = 0.5 * (self.buffer[current] + self.buffer[nxt]) * damping
            self._cursor = nxt
        return output


class BodyResonance:
    """Reduced-order modal body filter."""
    def __init__(self, sr=48000, modes=None):
        if sr <= 0:
            raise ValueError("sample rate must be positive")
        self.sr = int(sr)
        self.modes = modes if modes is not None else [
            [100, 5, 3], [200, 6, 2], [400, 8, 1],
            [800, 10, 0.5], [1600, 12, -1], [3200, 15, -3]
        ]

    def _biquad_peak(self, x, f0, Q, gain_db):
        if f0 <= 0 or Q <= 0 or f0 >= self.sr / 2:
            return np.asarray(x, dtype=np.float64).copy()
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * f0 / self.sr
        alpha = np.sin(w0) / (2 * Q)
        b0, b1, b2 = 1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A
        a0, a1, a2 = 1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A
        y = np.zeros_like(x, dtype=np.float64)
        x1 = x2 = y1 = y2 = 0.0
        for i, xi in enumerate(x):
            yi = (b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0
            x2, x1, y2, y1 = x1, xi, y1, yi
            y[i] = yi
        return y

    def __call__(self, x):
        y = np.asarray(x, dtype=np.float64).copy()
        if y.ndim != 1:
            raise ValueError("body input must be mono")
        for f0, Q, gain in self.modes:
            y = self._biquad_peak(y, f0, Q, gain)
        return y


class PhysicalGuitarModel:
    def __init__(self, sr=48000, f0=196.0):
        if sr <= 0 or f0 <= 0:
            raise ValueError("sr and f0 must be positive")
        self.sr = int(sr)
        self.string = DigitalWaveguideString(sr=sr, f0=f0)
        self.body = BodyResonance(sr=sr)

    def synthesize(self, n_samples, fret=0, pluck_pos=0.5, dynamics=1.0):
        self.string.pluck(fret=fret, pluck_pos=pluck_pos, dynamics=dynamics)
        raw = self.string.step(n_samples, fret=fret)
        return self.body(raw)
