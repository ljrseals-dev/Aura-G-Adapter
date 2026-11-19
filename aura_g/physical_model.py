"""Digital waveguide with fret-aware tuning, bounded damping, and blockwise control."""
import numpy as np


class DigitalWaveguideString:
    def __init__(self, sr=48000, f0=196.0, n_strings=6, fret_damping_scale=1.0):
        if (not isinstance(sr, (int, np.integer)) or sr <= 0 or
                not np.isfinite(f0) or f0 <= 0 or
                not isinstance(n_strings, (int, np.integer)) or n_strings <= 0 or
                not np.isfinite(fret_damping_scale) or fret_damping_scale < 0):
            raise ValueError("sr and n_strings must be positive integers; f0 finite positive; damping scale finite non-negative")
        self.sr, self.f0 = int(sr), float(f0)
        self.n_strings = int(n_strings)
        self.fret_damping_scale = float(fret_damping_scale)
        if self.f0 >= self.sr / 2:
            raise ValueError("string fundamental must be below Nyquist")
        self.delay = max(2, int(round(self.sr / self.f0)))
        self.buffer = np.zeros(self.delay, dtype=np.float64)
        self._cursor = 0

    def fret_damping(self, fret):
        if not isinstance(fret, (int, np.integer)) or fret < 0:
            raise ValueError("fret must be a non-negative integer")
        return float(np.clip(1.0 - 0.05 * fret * self.fret_damping_scale, 0.0, 1.0))

    def pluck(self, fret=0, pluck_pos=0.5, dynamics=1.0):
        if not isinstance(fret, (int, np.integer)) or fret < 0:
            raise ValueError("fret must be a non-negative integer")
        if (not np.isfinite(pluck_pos) or not 0 < pluck_pos < 1 or
                not np.isfinite(dynamics) or not 0 <= dynamics <= 1):
            raise ValueError("pluck_pos must be finite in (0, 1) and dynamics finite in [0, 1]")
        max_fret = int(np.floor(12.0 * np.log2((self.sr / 2.0) / self.f0)))
        if fret > max_fret:
            raise ValueError("fretted string fundamental must be below Nyquist")
        frequency = self.f0 * 2 ** (fret / 12.0)
        if not np.isfinite(frequency) or frequency >= self.sr / 2:
            raise ValueError("fretted string fundamental must be finite and below Nyquist")
        self.delay = max(2, int(round(self.sr / frequency)))
        idx = int(np.clip(round(pluck_pos * self.delay), 1, self.delay - 1))
        excitation = np.zeros(self.delay, dtype=np.float64)
        excitation[:idx] = np.linspace(0, 1, idx, endpoint=False)
        excitation[idx:] = np.linspace(1, 0, self.delay - idx)
        self.buffer = excitation * dynamics
        self._cursor = 0

    def step(self, n_samples, fret=0):
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 0:
            raise ValueError("n_samples must be a non-negative integer")
        if len(self.buffer) < 2:
            self.buffer = np.pad(self.buffer, (0, 2 - len(self.buffer)))
        output = np.empty(n_samples, dtype=np.float64)
        damping = self.fret_damping(fret)
        size = len(self.buffer)
        for index in range(n_samples):
            current = self._cursor
            following = (current + 1) % size
            output[index] = self.buffer[current]
            self.buffer[current] = 0.5 * (self.buffer[current] + self.buffer[following]) * damping
            self._cursor = following
        return output


class BodyResonance:
    """Reduced-order modal body filter with persistent per-mode IIR state."""
    def __init__(self, sr=48000, modes=None):
        if not isinstance(sr, (int, np.integer)) or sr <= 0:
            raise ValueError("sample rate must be a positive integer")
        self.sr = int(sr)
        self.modes = modes if modes is not None else [
            [100, 5, 3], [200, 6, 2], [400, 8, 1],
            [800, 10, 0.5], [1600, 12, -1], [3200, 15, -3]
        ]
        validated_modes = []
        for mode in self.modes:
            if len(mode) != 3:
                raise ValueError("each body mode must be (frequency, Q, gain_db)")
            f0, q, gain = map(float, mode)
            if not np.all(np.isfinite([f0, q, gain])) or f0 <= 0 or q <= 0:
                raise ValueError("body mode frequency and Q must be finite and positive; gain must be finite")
            validated_modes.append((f0, q, float(np.clip(gain, -36.0, 36.0))))
        self.modes = validated_modes
        self._states = [(0.0, 0.0, 0.0, 0.0) for _ in self.modes]

    def _biquad_peak(self, x, f0, q, gain_db, state):
        if f0 >= self.sr / 2:
            return np.asarray(x, dtype=np.float64).copy(), state
        amplitude = 10 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * f0 / self.sr
        alpha = np.sin(w0) / (2.0 * q)
        b0, b1, b2 = 1 + alpha * amplitude, -2 * np.cos(w0), 1 - alpha * amplitude
        a0, a1, a2 = 1 + alpha / amplitude, -2 * np.cos(w0), 1 - alpha / amplitude
        x1, x2, y1, y2 = state
        output = np.empty_like(x, dtype=np.float64)
        for index, sample in enumerate(x):
            value = (b0 * sample + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0
            x2, x1, y2, y1 = x1, sample, y1, value
            output[index] = value
        return output, (x1, x2, y1, y2)

    def __call__(self, x):
        raw = np.asarray(x)
        if np.iscomplexobj(raw):
            raise ValueError("body input must be real-valued")
        output = np.asarray(raw, dtype=np.float64).copy()
        if output.ndim != 1 or not np.all(np.isfinite(output)):
            raise ValueError("body input must be finite mono audio")
        for index, (f0, q, gain) in enumerate(self.modes):
            output, self._states[index] = self._biquad_peak(
                output, f0, q, gain, self._states[index])
        return output


class PhysicalGuitarModel:
    def __init__(self, sr=48000, f0=196.0):
        if not isinstance(sr, (int, np.integer)) or sr <= 0 or not np.isfinite(f0) or f0 <= 0:
            raise ValueError("sr must be a positive integer and f0 must be finite and positive")
        self.sr = int(sr)
        self.string = DigitalWaveguideString(sr=self.sr, f0=f0)
        self.body = BodyResonance(sr=self.sr)
        self._active_midi = None

    def synthesize(self, n_samples, fret=0, pluck_pos=0.5, dynamics=1.0):
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 0:
            raise ValueError("n_samples must be a non-negative integer")
        if (not isinstance(fret, (int, np.integer)) or fret < 0 or
                not np.all(np.isfinite([pluck_pos, dynamics])) or
                not 0 < pluck_pos < 1 or not 0 <= dynamics <= 1):
            raise ValueError("invalid fret, pluck position, or dynamics")
        self.string.pluck(fret=fret, pluck_pos=pluck_pos, dynamics=dynamics)
        raw = self.string.step(n_samples, fret=fret)
        return self.body(raw)

    def synthesize_sequence(self, buses, n_samples, gate=0.02, pluck_pos=0.3):
        """Render a sequence of latent pitch/dynamics controls without block retriggers."""
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 0:
            raise ValueError("n_samples must be a non-negative integer")
        if not np.isfinite(gate) or gate < 0 or not np.isfinite(pluck_pos) or not 0 < pluck_pos < 1:
            raise ValueError("gate and pluck position must be finite and in range")
        raw = np.empty(n_samples, dtype=np.float64)
        offset = 0
        for bus in buses:
            if bus.sample_rate != self.sr or not isinstance(bus.hop, (int, np.integer)) or bus.hop <= 0:
                raise ValueError("latent bus sample rate and hop must match the physical renderer")
            if offset >= n_samples:
                break
            gesture = bus.get_pitch_gesture()
            dynamics = float(gesture[6])
            confidence = bus.get_pitch_confidence()
            if dynamics >= gate and confidence >= 0.5:
                midi_value = 69.0 + 72.0 * float(gesture[0])
                if np.isfinite(midi_value) and 0.0 <= midi_value <= 127.0:
                    midi = int(round(midi_value))
                    onset = float(gesture[1]) > 0.5
                    frequency = 440.0 * 2.0 ** ((midi - 69) / 12.0)
                    if frequency < self.sr / 2:
                        if onset or midi != self._active_midi:
                            self.string.f0 = frequency
                            self.string.pluck(fret=0, pluck_pos=pluck_pos,
                                              dynamics=float(np.clip(dynamics, 0.0, 1.0)))
                            self._active_midi = midi
                    else:
                        # A valid MIDI pitch can still exceed Nyquist at a
                        # low renderer sample rate; treat it as an invalid
                        # control frame instead of raising from the DSP loop.
                        self._active_midi = None
                else:
                    self._active_midi = None
            else:
                self._active_midi = None
            count = min(int(bus.hop), n_samples - offset)
            raw[offset:offset + count] = self.string.step(count)
            offset += count
        if offset < n_samples:
            raw[offset:] = self.string.step(n_samples - offset)
        return self.body(raw)
