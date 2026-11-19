"""Joint note + fingering from latent bus."""
import numpy as np

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OPEN_STRINGS = [40, 45, 50, 55, 59, 64]


def hz_to_midi(f0):
    if not np.isfinite(f0) or f0 <= 0:
        return 0
    return int(round(69 + 12 * np.log2(f0 / 440.0)))


def midi_to_name(m):
    if not 0 <= int(m) <= 127:
        raise ValueError("MIDI note must be in [0, 127]")
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def fingering_from_midi(midi, max_fret=15):
    candidates = []
    for s, open_midi in enumerate(OPEN_STRINGS):
        fret = midi - open_midi
        if 0 <= fret <= max_fret:
            candidates.append((s, fret))
    if not candidates:
        return None
    s, f = min(candidates, key=lambda x: (abs(x[1] - 5) * 0.5 + x[0] * 0.1))
    return s, f


class TranscriptionEngine:
    """Monophonic note-event state machine; each note is emitted once with duration."""
    def __init__(self, sr=48000, hop=480, gate=0.02):
        if sr <= 0 or hop <= 0 or gate < 0:
            raise ValueError("invalid sample rate, hop, or gate")
        self.sr, self.hop, self.gate = int(sr), int(hop), float(gate)
        self.active_note = None
        self.frame_index = 0

    def _close_active(self, end_time):
        if self.active_note is not None:
            note = self.active_note
            note["end"] = max(float(end_time), note["start"])
            note["duration"] = note["end"] - note["start"]
            self.active_note = None
            return note
        return None

    def process(self, bus_history):
        events = []
        for bus in bus_history:
            pg = bus.get_pitch_gesture()
            # Zero log-pitch is the valid A4 (440 Hz) encoding; dynamics gates silence.
            f0 = 440.0 * (2 ** (float(pg[0]) * 6.0))
            onset = pg[1] > 0.5
            dynamics = float(pg[6])
            t = self.frame_index * self.hop / self.sr
            midi = hz_to_midi(f0) if f0 > 0 and dynamics >= self.gate else 0
            if self.active_note is not None and (midi == 0 or midi != self.active_note["midi"] or onset):
                closed = self._close_active(t)
                if closed is not None:
                    events.append(closed)
            if self.active_note is None and midi > 0:
                fing = fingering_from_midi(midi)
                self.active_note = {
                    "start": t, "midi": midi, "name": midi_to_name(midi),
                    "string": fing[0] if fing else -1, "fret": fing[1] if fing else -1,
                    "confidence": 1.0, "velocity": float(np.clip(dynamics, 0, 1)),
                }
            self.frame_index += 1
        return events

    def flush(self):
        """Close and return the final active note at end of stream."""
        end_time = self.frame_index * self.hop / self.sr
        note = self._close_active(end_time)
        return [note] if note is not None else []
