"""Karplus-Strong guitar synthesis and defensive Standard MIDI File Type-0 I/O."""
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
    frequency = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    count = int(fs * dur)
    period = max(2, int(round(fs / frequency)))
    noise = np.random.default_rng(seed).uniform(-1.0, 1.0, period)
    output = np.zeros(count, dtype=np.float64)
    initial = min(period, count)
    output[:initial] = noise[:initial] * (vel ** 1.5)
    for i in range(period, count):
        output[i] = 0.5 * (output[i - period] + output[i - period - 1]) * 0.996
    return output


def _vlq(value):
    """Encode a MIDI Variable Length Quantity (maximum 0x0FFFFFFF)."""
    if not isinstance(value, (int, np.integer)) or not 0 <= value <= 0x0FFFFFFF:
        raise ValueError("MIDI delta time must be an integer in [0, 0x0FFFFFFF]")
    result = [value & 0x7F]
    value >>= 7
    while value:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(result))


def _read_vlq(data, offset, end=None):
    """Read a bounded MIDI VLQ and reject truncation/overlong values."""
    limit = len(data) if end is None else min(len(data), end)
    value = 0
    for _ in range(4):
        if offset >= limit:
            raise ValueError("truncated MIDI variable-length quantity")
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, offset
    raise ValueError("MIDI variable-length quantity exceeds four bytes")


def _build_midi_bytes(notes, bpm=120, tpq=480):
    try:
        bpm_value = float(bpm)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("bpm must be a finite positive number") from None
    if not np.isfinite(bpm_value) or bpm_value <= 0:
        raise ValueError("bpm must be a finite positive number")
    tempo_value = 60e6 / bpm_value
    if not np.isfinite(tempo_value) or not 0.5 <= tempo_value < 0xFFFFFF + 0.5:
        raise ValueError("bpm is outside the Standard MIDI 24-bit tempo range")
    tempo_us = int(round(tempo_value))
    if not 1 <= tempo_us <= 0xFFFFFF:
        raise ValueError("bpm is outside the Standard MIDI 24-bit tempo range")
    if not isinstance(tpq, (int, np.integer)) or not 1 <= tpq <= 0x7FFF:
        raise ValueError("tpq must be in [1, 32767]")
    tpq = int(tpq)
    events = []
    for note in notes:
        if not np.isfinite(note.start) or not np.isfinite(note.dur) or note.start < 0 or note.dur <= 0:
            raise ValueError("note start must be non-negative and duration must be positive")
        if not isinstance(note.midi, (int, np.integer)) or not 0 <= note.midi <= 127:
            raise ValueError("note MIDI pitch must be in [0, 127]")
        if not isinstance(note.vel, (int, np.integer)) or not 0 <= note.vel <= 127:
            raise ValueError("note velocity must be in [0, 127]")
        try:
            start_seconds, duration_seconds = float(note.start), float(note.dur)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("note start and duration must be finite numbers") from None
        if (not np.isfinite(start_seconds) or not np.isfinite(duration_seconds) or
                start_seconds < 0 or duration_seconds <= 0):
            raise ValueError("note start must be non-negative and duration must be positive")
        start_tick = int(round(start_seconds * tpq * bpm_value / 60.0))
        end_tick = max(start_tick + 1, int(round((start_seconds + duration_seconds) * tpq * bpm_value / 60.0)))
        events.extend([(start_tick, 1, bytes([0x90, note.midi, note.vel])),
                       (end_tick, 0, bytes([0x80, note.midi, 0]))])
    events.sort(key=lambda event: (event[0], event[1]))
    track = bytearray(_vlq(0) + bytes([0xFF, 0x51, 3]) + tempo_us.to_bytes(3, "big"))
    previous_tick = 0
    for tick, _, message in events:
        track.extend(_vlq(tick - previous_tick))
        track.extend(message)
        previous_tick = tick
    track.extend(_vlq(0) + bytes([0xFF, 0x2F, 0]))
    return b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpq) + b"MTrk" + struct.pack(">I", len(track)) + track


def write_midi(path, notes, bpm=120, tpq=480):
    with open(path, "wb") as stream:
        stream.write(_build_midi_bytes(notes, bpm=bpm, tpq=tpq))


def _read_midi_impl(data):
    """Parse bounded SMF format 0/1 data with a merged, tick-based tempo map."""
    data = bytes(data)
    if len(data) < 14 or data[:4] != b"MThd":
        raise ValueError("invalid or truncated MIDI header")
    header_length, file_format, track_count, division = struct.unpack(">IHHH", data[4:14])
    if header_length < 6 or len(data) < 8 + header_length:
        raise ValueError("invalid MIDI header length")
    if file_format not in (0, 1) or track_count < 1 or (file_format == 0 and track_count != 1):
        raise ValueError("unsupported MIDI format or track count")
    if division == 0 or division & 0x8000:
        raise ValueError("SMPTE or zero MIDI time division is unsupported")

    # Keep absolute-tick events until every track has been read. Format-1
    # tempo events are global; converting each track while parsing would
    # incorrectly apply a tempo change only to later events in that track.
    merged = []
    position = 8 + header_length
    for track_index in range(track_count):
        if position + 8 > len(data) or data[position:position + 4] != b"MTrk":
            raise ValueError("missing or truncated MIDI track")
        length = struct.unpack(">I", data[position + 4:position + 8])[0]
        offset, end = position + 8, position + 8 + length
        if end > len(data):
            raise ValueError("MIDI track extends beyond file")
        tick, running_status, sequence = 0, None, 0
        saw_end_of_track = False
        while offset < end:
            delta, offset = _read_vlq(data, offset, end)
            tick += delta
            if offset >= end:
                raise ValueError("truncated MIDI event")
            status = data[offset]
            if status == 0xFF:
                offset += 1
                if offset >= end:
                    raise ValueError("truncated MIDI meta event")
                kind = data[offset]
                offset += 1
                size, offset = _read_vlq(data, offset, end)
                if offset + size > end:
                    raise ValueError("MIDI meta event exceeds track")
                payload = data[offset:offset + size]
                if kind == 0x51:
                    if size != 3:
                        raise ValueError("MIDI tempo meta event must contain exactly three bytes")
                    tempo = int.from_bytes(payload, "big")
                    if tempo == 0:
                        raise ValueError("MIDI tempo cannot be zero")
                    merged.append((tick, 0, track_index, sequence, "tempo", tempo))
                offset += size
                sequence += 1
                if kind == 0x2F:
                    if size != 0:
                        raise ValueError("MIDI end-of-track event must be empty")
                    if offset != end:
                        raise ValueError("data follows MIDI end-of-track event")
                    saw_end_of_track = True
                    break
                continue
            if status in (0xF0, 0xF7):
                offset += 1
                size, offset = _read_vlq(data, offset, end)
                if offset + size > end:
                    raise ValueError("MIDI SysEx event exceeds track")
                offset += size
                running_status = None
                sequence += 1
                continue
            if status & 0x80:
                running_status = status
                offset += 1
            elif running_status is None:
                raise ValueError("MIDI running status used before channel status")
            status = running_status
            if not 0x80 <= status <= 0xEF:
                raise ValueError("unsupported MIDI system event")
            command = status & 0xF0
            data_bytes = 1 if command in (0xC0, 0xD0) else 2
            if offset + data_bytes > end:
                raise ValueError("truncated MIDI channel event")
            first = data[offset]
            second = data[offset + 1] if data_bytes == 2 else 0
            offset += data_bytes
            if first > 127 or second > 127:
                raise ValueError("invalid MIDI data byte")
            merged.append((tick, 1, track_index, sequence, "channel",
                           (status & 0x0F, command, first, second)))
            sequence += 1
        if not saw_end_of_track:
            raise ValueError("MIDI track is missing its end-of-track event")
        position = end
    if position != len(data):
        raise ValueError("unexpected data after declared MIDI tracks")

    merged.sort(key=lambda event: event[:4])
    tempo, previous_tick, elapsed = 500000, 0, 0.0
    notes, open_notes = [], {}
    for tick, _, track_index, _, kind, payload in merged:
        elapsed += (tick - previous_tick) * tempo / 1e6 / division
        previous_tick = tick
        if kind == "tempo":
            tempo = payload
            continue
        channel, command, pitch, velocity = payload
        key = (track_index, channel, pitch)
        if command == 0x90 and velocity > 0:
            # Repeated same-pitch note-ons may overlap on one track/channel.
            # Preserve every voice; pair note-offs with the oldest open voice.
            open_notes.setdefault(key, []).append((elapsed, velocity))
        elif command == 0x80 or (command == 0x90 and velocity == 0):
            active = open_notes.get(key)
            if active:
                start_time, start_velocity = active.pop(0)
                notes.append(Note(start_time, max(0.0, elapsed - start_time),
                                  pitch, start_velocity))
                if not active:
                    del open_notes[key]
    return sorted(notes, key=lambda note: (note.start, note.midi))

def read_midi(path):
    with open(path, "rb") as stream:
        return _read_midi_impl(stream.read())


if __name__ == "__main__":
    import os
    import tempfile
    sample_notes = [Note(0.0, 1.0, 60, 100), Note(0.5, 0.8, 64, 90), Note(1.0, 1.5, 67, 80)]
    midi_path = os.path.join(tempfile.gettempdir(), "test_guitar.mid")
    write_midi(midi_path, sample_notes)
    recovered = read_midi(midi_path)
    assert len(recovered) == len(sample_notes)
    print("MIDI round-trip OK:", os.path.getsize(midi_path), "bytes")
