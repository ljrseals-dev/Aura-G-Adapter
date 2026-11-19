"""Validated Kelly-Lochbaum vocal-tract filter for talk-box-style effects."""
import numpy as np
from scipy import signal

VOW_CMAJOR = {
    "ah": (1.8, 1.6, 1.4, 0.9, 0.7, 0.8, 1.2, 1.0),
    "eh": (1.2, 1.1, 1.0, 0.6, 0.5, 0.9, 1.3, 1.1),
    "ee": (1.0, 0.9, 0.8, 0.5, 0.3, 1.1, 1.5, 1.2),
    "oh": (1.4, 1.3, 1.2, 1.1, 0.9, 0.7, 0.8, 0.4),
    "oo": (1.2, 1.0, 0.9, 1.0, 0.4, 0.6, 0.7, 0.2),
}
ORDER = ["ah", "eh", "ee", "oh", "oo"]
VOW = {
    "ah": (3.498, 0.1, 8.0, 0.515, 8.0, 0.422, 8.0, 0.984),
    "eh": (0.684, 0.864, 0.691, 0.1, 0.956, 2.821, 0.978, 0.989),
    "ee": (0.683, 0.907, 0.686, 0.1, 0.1, 8.0, 0.62, 0.273),
    "oh": (4.532, 0.1, 0.448, 1.095, 0.1, 0.231, 1.78, 0.917),
    "oo": (2.39, 8.0, 0.998, 4.118, 0.457, 0.493, 8.0, 0.1),
}
N = 48


def area_function(vowel, jaw=1.0, tongue=1.0, lips=1.0):
    """Map vowel position [0,1] and positive articulator scales to 48 areas."""
    controls = np.asarray([vowel, jaw, tongue, lips], dtype=np.float64)
    if not np.all(np.isfinite(controls)):
        raise ValueError("vowel and articulator controls must be finite")
    if not 0.0 <= vowel <= 1.0 or min(jaw, tongue, lips) <= 0:
        raise ValueError("vowel must be in [0,1] and articulator scales must be positive")
    position = float(vowel) * 4.0
    lower = min(int(position), 3)
    fraction = position - lower
    lower_area = np.asarray(VOW[ORDER[lower]], dtype=np.float64)
    upper_area = np.asarray(VOW[ORDER[lower + 1]], dtype=np.float64)
    areas = np.exp((1.0 - fraction) * np.log(lower_area) +
                   fraction * np.log(upper_area))
    areas[0] *= jaw
    areas[1] *= jaw
    areas[4] *= tongue
    areas[7] *= lips
    x8 = (np.arange(8) + 0.5) / 8.0
    x48 = (np.arange(N) + 0.5) / N
    return np.maximum(np.interp(x48, x8, areas), 0.05)


class Tract:
    """Stateful 48-section Kelly-Lochbaum tract.

    Vowel controls are sampled at the start of each block-sized control
    interval; wave propagation itself runs at the audio sample rate.
    """
    def __init__(self, rg=0.75, rl=-0.9):
        if not np.all(np.isfinite([rg, rl])) or not -1.0 < rg < 1.0 or not -1.0 < rl < 1.0:
            raise ValueError("glottal and lip reflection coefficients must be finite and inside (-1,1)")
        self.f = np.zeros(N, dtype=np.float64)
        self.b = np.zeros(N, dtype=np.float64)
        self.rg, self.rl = float(rg), float(rl)
        self.prev = 0.0

    def process(self, src, vowel_curve, block=32):
        """Process finite mono excitation and a matching per-sample vowel curve."""
        if not isinstance(block, (int, np.integer)) or block <= 0:
            raise ValueError("block must be a positive integer")
        raw_src, raw_curve = np.asarray(src), np.asarray(vowel_curve)
        if np.iscomplexobj(raw_src) or np.iscomplexobj(raw_curve):
            raise ValueError("source and vowel curve must be real-valued")
        source = np.asarray(raw_src, dtype=np.float64)
        curve = np.asarray(raw_curve, dtype=np.float64)
        if (source.ndim != 1 or curve.ndim != 1 or source.shape != curve.shape or
                not np.all(np.isfinite(source)) or not np.all(np.isfinite(curve))):
            raise ValueError("source and vowel curve must be matching finite mono arrays")
        if np.any((curve < 0.0) | (curve > 1.0)):
            raise ValueError("vowel curve values must be in [0,1]")
        if len(source) == 0:
            return np.zeros(0, dtype=np.float64)

        output = np.empty(len(source), dtype=np.float64)
        forward, backward = self.f, self.b
        for start in range(0, len(source), int(block)):
            areas = area_function(curve[start])
            reflection = (areas[:-1] - areas[1:]) / (areas[:-1] + areas[1:])
            stop = min(start + int(block), len(source))
            for index in range(start, stop):
                next_forward = np.empty(N, dtype=np.float64)
                next_backward = np.empty(N, dtype=np.float64)
                next_forward[0] = source[index] + self.rg * backward[0]
                next_forward[1:] = (1.0 + reflection) * forward[:-1] - reflection * backward[1:]
                next_backward[:-1] = reflection * forward[:-1] + (1.0 - reflection) * backward[1:]
                next_backward[-1] = self.rl * forward[-1]
                forward, backward = next_forward, next_backward
                lip_pressure = (1.0 + self.rl) * forward[-1]
                output[index] = lip_pressure - 0.97 * self.prev
                self.prev = lip_pressure
        self.f, self.b = forward, backward
        return output


def formants(vowel, fs=48000, n=2048, fmax=4200, k=3):
    """Estimate the first k spectral peaks of the tract impulse response."""
    if (not isinstance(fs, (int, np.integer)) or fs <= 0 or
            not isinstance(n, (int, np.integer)) or n < 4 or
            not isinstance(k, (int, np.integer)) or k < 0 or
            not np.isfinite(fmax) or not 0 < fmax <= fs / 2):
        raise ValueError("fs and n must be positive integers; k and fmax must be valid")
    impulse = np.zeros(int(n), dtype=np.float64)
    impulse[0] = 1.0
    response = Tract().process(impulse, np.full(int(n), vowel), block=min(32, int(n)))
    windowed = response * np.hanning(int(n))
    n_fft = max(1 << 15, 1 << (int(n) - 1).bit_length())
    spectrum = np.abs(np.fft.rfft(windowed, n_fft))
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / int(fs))
    mask = frequencies <= fmax
    limited = spectrum[mask]
    if k == 0 or limited.size < 3 or not np.any(limited):
        return []
    peaks, _ = signal.find_peaks(limited, prominence=0.05 * float(limited.max()))
    return [round(float(frequencies[mask][peak])) for peak in peaks[:int(k)]]
