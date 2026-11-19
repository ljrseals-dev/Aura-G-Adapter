"""Blind effect extraction: estimate an FX transfer from dry/wet pairs."""
import numpy as np


class BlindExtractor:
    """Frequency-domain Wiener-style inversion for approximately LTI effects."""
    def __init__(self, sr=48000, n_fft=4096, hop=1024, reg=1e-3):
        if (not isinstance(sr, (int, np.integer)) or sr <= 0 or
                not isinstance(n_fft, (int, np.integer)) or n_fft < 2 or
                not isinstance(hop, (int, np.integer)) or hop <= 0 or
                not np.isfinite(reg) or reg <= 0):
            raise ValueError("sr, n_fft, and hop must be positive integers; regularization must be finite and positive")
        self.sr, self.n_fft, self.hop, self.reg = int(sr), int(n_fft), int(hop), float(reg)
        self.H = None

    def _stft(self, x):
        raw = np.asarray(x)
        if np.iscomplexobj(raw):
            raise ValueError("audio must be real-valued")
        x = np.asarray(raw, dtype=np.float64)
        if x.ndim != 1 or not np.all(np.isfinite(x)):
            raise ValueError("audio must be a finite mono array")
        if len(x) < self.n_fft:
            return np.empty((0, self.n_fft // 2 + 1), dtype=np.complex128)
        win = np.hanning(self.n_fft)
        return np.asarray([np.fft.rfft(x[i:i + self.n_fft] * win)
                           for i in range(0, len(x) - self.n_fft + 1, self.hop)])

    def fit(self, dry, wet):
        dry, wet = np.asarray(dry), np.asarray(wet)
        if dry.ndim != 1 or wet.ndim != 1 or len(dry) != len(wet):
            raise ValueError("dry and wet must be mono arrays of equal length")
        D, W = self._stft(dry), self._stft(wet)
        n = min(len(D), len(W))
        if n == 0:
            raise ValueError("signals are shorter than n_fft; no analysis frames available")
        D, W = D[:n], W[:n]
        num = np.sum(W * np.conj(D), axis=0)
        den = np.sum(np.abs(D) ** 2, axis=0) + self.reg
        self.H = num / den
        return self.H.copy()

    def apply(self, x):
        if self.H is None:
            raise RuntimeError("Call fit() first.")
        x = np.asarray(x, dtype=np.float64)
        X = self._stft(x)
        if len(x) == 0:
            return np.zeros(0, dtype=np.float64)
        if len(X) == 0:
            # The learned transfer estimate has no full analysis frame to
            # operate on; bypass instead of replacing a short clip with silence.
            return x.copy()
        if X.shape[1] != len(self.H):
            raise ValueError("stored transfer function does not match n_fft")
        out = np.zeros(len(x), dtype=np.float64)
        norm = np.zeros(len(x), dtype=np.float64)
        win = np.hanning(self.n_fft)
        for i, frame in enumerate(X):
            start = i * self.hop
            end = min(start + self.n_fft, len(x))
            frame_out = np.fft.irfft(frame * self.H, n=self.n_fft) * win
            out[start:end] += frame_out[:end - start]
            norm[start:end] += win[:end - start] ** 2
        valid = norm > 1e-10
        out[valid] /= norm[valid]
        return out
