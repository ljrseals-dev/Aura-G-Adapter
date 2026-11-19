"""Shared Latent Control Bus — the core abstraction of AURA-G."""
import numpy as np

BUS_DIM = 32
HOP_SECONDS = 0.01  # 100 Hz update rate

PITCH_GESTURE = slice(0, 8)
TIMBRE = slice(8, 16)
EXPRESSION = slice(16, 24)
TRANSCRIPTION = slice(24, 28)
RESERVED = slice(28, 32)


class LatentBus:
    """Validated 32-dimensional control vector with copy-only public access."""
    def __init__(self, vector=None, sample_rate=48000, hop=480):
        if (not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0 or
                not isinstance(hop, (int, np.integer)) or hop <= 0):
            raise ValueError("sample_rate and hop must be positive integers")
        if vector is None:
            vector = np.zeros(BUS_DIM, dtype=np.float32)
        raw = np.asarray(vector)
        if np.iscomplexobj(raw):
            raise ValueError("latent vector must be real-valued")
        values = np.asarray(raw, dtype=np.float32)
        if values.shape != (BUS_DIM,) or not np.all(np.isfinite(values)):
            raise ValueError("latent vector must be finite and have exactly 32 elements")
        self.sample_rate, self.hop = int(sample_rate), int(hop)
        self._vector = values.copy()

    @property
    def vector(self):
        """Return a snapshot; mutation must go through validated write methods."""
        return self._vector.copy()

    @staticmethod
    def _values(values, count, label):
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            raise ValueError(f"{label} must be real-valued")
        result = np.asarray(raw, dtype=np.float32)
        if result.shape != (count,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{label} must contain {count} finite values")
        return result

    def write_pitch_gesture(self, f0_log_hz, onset, bend, vib_rate, vib_depth,
                            pluck_pos, dynamics, palm_mute):
        self._vector[PITCH_GESTURE] = self._values(
            [f0_log_hz, onset, bend, vib_rate, vib_depth,
             pluck_pos, dynamics, palm_mute], 8, "pitch gesture")

    def write_timbre(self, centroid, rolloff, flatness, zcr,
                     harmonic_ratio, even_odd, attack_slope, sustain):
        self._vector[TIMBRE] = self._values(
            [centroid, rolloff, flatness, zcr,
             harmonic_ratio, even_odd, attack_slope, sustain], 8, "timbre")

    def write_expression(self, macros: np.ndarray):
        self._vector[EXPRESSION] = self._values(macros, 8, "expression macros")

    def write_transcription(self, note_id, string, fret, confidence):
        self._vector[TRANSCRIPTION] = self._values(
            [note_id, string, fret, confidence], 4, "transcription")

    def read(self):
        return self._vector.copy()

    def get_pitch_gesture(self):
        return self._vector[PITCH_GESTURE].copy()

    def get_timbre(self):
        return self._vector[TIMBRE].copy()

    def get_expression(self):
        return self._vector[EXPRESSION].copy()

    def get_transcription(self):
        return self._vector[TRANSCRIPTION].copy()
