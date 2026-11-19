"""Karplus-Strong guitar synthesis and defensive Standard MIDI File Type-0 I/O."""
import math
import struct
from dataclasses import dataclass
import numpy as np


@dataclass
class Note:
    start: float
    dur: float
    midi: int
    vel: int = 90


def pluck(midi: int, dur: float, vel: float, fs: int, seed=None) -> np.ndarray:
    """Synthesize a reproducible plucked string note."""
    if not isinstance(midi, (int, np.integer)) or not 0 <= midi <= 127:
        raise ValueError("midi must be an integer in [0, 127]")
    if not np.isfinite(dur) or dur < 0 or not np.isfinite(vel) or not 0 <= vel <= 1:
        raise ValueError("dur must be non-negative and vel must be in [0, 1]")
    if not isinstance(fs, (int, np.integer)) or fs <= 0:
        raise ValueError("fs must be a positive integer")
    freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    n_samples = int(fs * dur)
    period = max(2, int(round(fs / freq)))
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-1.0, 1.0, period)
    out = np.zeros(n_samples, dtype=np.float64)
    initial = min(period, n_samples)
    out[:initial] = noise[:initial] * (vel ** 1.5)
    for i in range(period, n_samples):
        out[i] = 0.5 * (out[i - period] + out[i - period - 1]) * 0.996
    return out


def _vlq(v):
    """Encode a MIDI Variable Length Quantity (maximum 0x0FFFFFFF)."""
    if not isinstance(v, (int, np.integer)) or not 0 <= v <= 0x0FFFFFFF:
        raise ValueError("MIDI delta time must be an integer in [0, 0x0FFFFFFF]")
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    return bytes(reversed(out))


def _read_vlq(data, q, end=None):
    """Read a bounded MIDI VLQ and reject truncation/overlong values."""
    limit = len(data) if end is None else min(len(data), end)
    value = 0
    for _ in range(4):
        if q >= limit:
            raise ValueError("truncated MIDI variable-length quantity")
        byte = data[q]
        q += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, q
    raise ValueError("MIDI variable-length quantity exceeds four bytes")


def _build_midi_bytes(notes, bpm=120, tpq=480):
    if not np.isfinite(bpm) or bpm <= 0 or not isinstance(tpq, int) or not 1 <= tpq <= 0x7FFF:
        raise ValueError("bpm must be positive and tpq must be in [1, 32767]")
    events = []
    for note in notes:
        if not np.isfinite(note.start) or not np.isfinite(note.dur) or note.start < 0 or note.dur < 0:
            raise ValueError("note start and duration must be finite and non-negative")
        if not isinstance(note.midi, (int, np.integer)) or not 0 <= note.midi <= 127:
            raise ValueError("note MIDI pitch must be in [0, 127]")
        if not isinstance(note.vel, (int, np.integer)) or not 0 <= note.vel <= 127:
            raise ValueError("note velocity must be in [0, 127]")
        t0 = int(round(note.start * tpq * bpm / 60))
        t1 = int(round((note.start + note.dur) * tpq * bpm / 60))
        events.extend([(t0, 1, bytes([0x90, note.midi, note.vel])),
                       (t1, 0, bytes([0x80, note.midi, 0]))])
    events.sort(key=lambda event: (event[0], event[1]))
    track = bytearray(_vlq(0) + bytes([0xFF, 0x51, 3]) + struct.pack(">I", int(60e6 / bpm))[1:])
    last_tick = 0
    for tick, _, message in events:
        track.extend(_vlq(tick - last_tick))
        track.extend(message)
        last_tick = tick
    track.extend(_vlq(0) + bytes([0xFF, 0x2F, 0]))
    return b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpq) + b"MTrk" + struct.pack(">I", len(track)) + track


def write_midi(path, notes, bpm=120, tpq=480):
    data = _build_midi_bytes(notes, bpm=bpm, tpq=tpq)
    with open(path, "wb") as stream:
        stream.write(data)


def _read_midi_impl(data):
    """Parse a bounded SMF file; malformed or unsupported files raise ValueError."""
    data = bytes(data)
    if len(data) < 14 or data[:4] != b"MThd":
        raise ValueError("invalid or truncated MIDI header")
    header_len, fmt, n_tracks, division = struct.unpack(">IHHH", data[4:14])
    if header_len < 6 or len(data) < 8 + header_len:
        raise ValueError("invalid MIDI header length")
    if fmt not in (0, 1) or n_tracks < 1 or (fmt == 0 and n_tracks != 1):
        raise ValueError("unsupported MIDI format or track count")
    if division == 0 or division & 0x8000:
        raise ValueError("SMPTE or zero MIDI time division is unsupported")
    pos = 8 + header_len
    tempo = 500000
    notes, open_notes = [], {}
    for _ in range(n_tracks):
        if pos + 8 > len(data) or data[pos:pos + 4] != b"MTrk":
            raise ValueError("missing or truncated MIDI track")
        length = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        q, end = pos + 8, pos + 8 + length
        if end > len(data):
            raise ValueError("MIDI track extends beyond file")
        tick, running_status = 0, None
        while q < end:
            delta, q = _read_vlq(data, q, end)
            tick += delta
            if q >= end:
                raise ValueError("truncated MIDI event")
            status = data[q]
            if status == 0xFF:
                q += 1
                if q + 2 > end:
                    raise ValueError("truncated MIDI meta event")
                kind, size = data[q], data[q + 1]
                q += 2
                if q + size > end:
                    raise ValueError("MIDI meta event exceeds track")
                if kind == 0x51 and size == 3:
                    tempo = int.from_bytes(data[q:q + 3], "big")
                    if tempo == 0:
                        raise ValueError("MIDI tempo cannot be zero")
                q += size
                continue
            if status in (0xF0, 0xF7):
                q += 1
                if q >= end:
                    raise ValueError("truncated MIDI SysEx event")
                size, q = _read_vlq(data, q, end)
                if q + size > end:
                    raise ValueError("MIDI SysEx event exceeds track")
                q += size
                running_status = None
                continue
            if status & 0x80:
                running_status = status
                q += 1
            elif running_status is None:
                raise ValueError("MIDI running status used before channel status")
            status = running_status
            count = 1 if (status & 0xF0) in (0xC0, 0xD0) else 2
            if q + count > end:
                raise ValueError("truncated MIDI channel event")
            first = data[q]
            second = data[q + 1] if count == 2 else 0
            q += count
            if first > 127 or second > 127:
                raise ValueError("invalid MIDI data byte")
            channel, command = status & 0x0F, status & 0xF0
            seconds = tick * tempo / 1e6 / division
            key = (channel, first)
            if command == 0x90 and second > 0:
                open_notes[key] = (seconds, second)
            elif command == 0x80 or (command == 0x90 and second == 0):
                if key in open_notes:
                    start, velocity = open_notes.pop(key)
                    notes.append(Note(start, max(0.0, seconds - start), first, velocity))
        pos = end
    return sorted(notes, key=lambda note: (note.start, note.midi))


def read_midi(path):
    with open(path, "rb") as stream:
        return _read_midi_impl(stream.read())


if __name__ == "__main__":
    import os
    import tempfile
    test_notes = [Note(0.0, 1.0, 60, 100), Note(0.5, 0.8, 64, 90), Note(1.0, 1.5, 67, 80)]
    path = os.path.join(tempfile.gettempdir(), "test_guitar.mid")
    write_midi(path, test_notes)
    got = read_midi(path)
    assert len(got) == len(test_notes)
    print("MIDI round-trip OK:", os.path.getsize(path), "bytes")
