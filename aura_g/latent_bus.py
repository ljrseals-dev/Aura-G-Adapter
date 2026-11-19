"""Shared Latent Control Bus — the core abstraction of AURA-G."""
from dataclasses import dataclass, field
import numpy as np

BUS_DIM = 32
HOP_SECONDS = 0.01  # 100 Hz update rate

PITCH_GESTURE = slice(0, 8)
TIMBRE = slice(8, 16)
EXPRESSION = slice(16, 24)
TRANSCRIPTION = slice(24, 28)
RESERVED = slice(28, 32)


@dataclass
class LatentBus:
    """Validated 32-dimensional control vector shared across AURA-G modules."""
    vector: np.ndarray = field(default_factory=lambda: np.zeros(BUS_DIM, dtype=np.float32))
    sample_rate: int = 48000
    hop: int = 480

    def __post_init__(self):
        if (not isinstance(self.sample_rate, (int, np.integer)) or self.sample_rate <= 0 or
                not isinstance(self.hop, (int, np.integer)) or self.hop <= 0):
            raise ValueError("sample_rate and hop must be positive integers")
        vector = np.asarray(self.vector, dtype=np.float32)
        if vector.shape != (BUS_DIM,) or not np.all(np.isfinite(vector)):
            raise ValueError("latent vector must be finite and have exactly 32 elements")
        self.sample_rate, self.hop = int(self.sample_rate), int(self.hop)
        self.vector = vector.copy()

    @staticmethod
    def _values(values, count, label):
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (count,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{label} must contain {count} finite values")
        return result

    def write_pitch_gesture(self, f0_log_hz, onset, bend, vib_rate, vib_depth,
                            pluck_pos, dynamics, palm_mute):
        self.vector[PITCH_GESTURE] = self._values(
            [f0_log_hz, onset, bend, vib_rate, vib_depth,
             pluck_pos, dynamics, palm_mute], 8, "pitch gesture")

    def write_timbre(self, centroid, rolloff, flatness, zcr,
                     harmonic_ratio, even_odd, attack_slope, sustain):
        self.vector[TIMBRE] = self._values(
            [centroid, rolloff, flatness, zcr,
             harmonic_ratio, even_odd, attack_slope, sustain], 8, "timbre")

    def write_expression(self, macros: np.ndarray):
        self.vector[EXPRESSION] = self._values(macros, 8, "expression macros")

    def write_transcription(self, note_id, string, fret, confidence):
        self.vector[TRANSCRIPTION] = self._values(
            [note_id, string, fret, confidence], 4, "transcription")

    def read(self):
        return self.vector.copy()

    def get_pitch_gesture(self):
        return self.vector[PITCH_GESTURE].copy()

    def get_timbre(self):
        return self.vector[TIMBRE].copy()

    def get_expression(self):
        return self.vector[EXPRESSION].copy()

    def get_transcription(self):
        return self.vector[TRANSCRIPTION].copy()
