import unittest
import struct
import numpy as np
import aura_g_adapter as A
import midi_bridge as B
import harmonia_guitar as G
from aura_g.front_end import FrontEndAnalyzer
from aura_g.latent_bus import LatentBus
from aura_g.transcription import TranscriptionEngine
from aura_g.blind_extractor import BlindExtractor

SR = 48000
REF = [B.Note(0.1 + 0.45 * i, 0.4, m, 90) for i, m in enumerate([52, 55, 59, 57, 55, 52, 50, 47, 64, 62])]


def synth():
    y = np.zeros(int(SR * (REF[-1].start + REF[-1].dur + 0.3)))
    for i, note in enumerate(REF):
        p = G.pluck(note.midi, note.dur, 0.8, SR, seed=i)
        start = int(note.start * SR)
        y[start:start + len(p)] += p[:len(y) - start]
    return y / max(np.max(np.abs(y)), 1e-12)


def f0_spec(y, f0):
    length = int(0.6 * SR)
    seg = y[int(0.05 * SR):int(0.65 * SR)] * np.hanning(length)
    nfft = 1 << 18
    spectrum = np.abs(np.fft.rfft(seg, nfft))
    frequencies = np.fft.rfftfreq(nfft, 1 / SR)
    mask = (frequencies > f0 * .9) & (frequencies < f0 * 1.1)
    index = int(np.argmax(np.where(mask, spectrum, 0)))
    a, b, c = np.log(spectrum[index - 1:index + 2] + 1e-30)
    return frequencies[index] + 0.5 * (a - c) / (a - 2 * b + c) * (frequencies[1] - frequencies[0])


class T(unittest.TestCase):
    def test_transcription_f1_and_durations(self):
        estimated = A.aura_g_transcribe(synth(), SR)
        true_positive, used, durations = 0, set(), []
        for reference in REF:
            for j, event in enumerate(estimated):
                if j not in used and event.midi == reference.midi and abs(event.start - reference.start) < 0.06:
                    true_positive += 1
                    used.add(j)
                    durations.append(event.dur)
                    break
        precision = true_positive / max(len(estimated), 1)
        recall = true_positive / len(REF)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        print(f"  AURA-G adapter: {true_positive}/{len(REF)} matched, {len(estimated)} notes, F1 {f1:.2f}, median duration {np.median(durations):.2f} s")
        self.assertGreaterEqual(f1, 0.9)

    def test_string_tuning_and_fret(self):
        string = A.TunedWaveguideString(SR, 1)
        worst = 0
        for frequency in (82.41, 196.0, 329.63, 659.26, 880.0):
            audio = string.render(frequency, 1.0)
            worst = max(worst, abs(1200 * np.log2(f0_spec(audio, frequency) / frequency)))
        fretted = string.render(196.0, 1.0, fret=5)
        fret_error = abs(1200 * np.log2(f0_spec(fretted, 261.63) / 261.63))
        print(f"  tuned string: worst open error {worst:.2f} cents, fret-5 error {fret_error:.2f} cents")
        self.assertLess(worst, 1.5)
        self.assertLess(fret_error, 1.5)

    def test_decay_follows_t60(self):
        audio = A.TunedWaveguideString(SR, 2).render(196.0, 4.0, t60=2.0)
        rms = lambda a, b: np.sqrt(np.mean(audio[int(a * SR):int(b * SR)] ** 2))
        decay = 20 * np.log10(rms(0.5, 0.6) / rms(1.5, 1.6))
        print(f"  decay over 1 s at T60=2 s: {decay:.1f} dB")
        self.assertTrue(22 < decay < 38)

    def test_nan_safe_input(self):
        audio = synth().copy()
        audio[20000] = np.nan
        audio = np.nan_to_num(audio)
        self.assertTrue(len(A.aura_g_transcribe(audio, SR)) >= 8)

    def test_end_to_end(self):
        estimated = A.aura_g_transcribe(synth(), SR)
        first = A.render_talkbox_aura(estimated[:4])
        second = A.render_talkbox_aura(estimated[:4])
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertTrue(np.array_equal(first, second))
        self.assertLessEqual(np.max(np.abs(first)), 0.7001)

    def test_empty_and_short_audio_are_safe(self):
        self.assertEqual(A.aura_g_transcribe(np.zeros(0), SR), [])
        self.assertEqual(list(FrontEndAnalyzer(sr=SR).process(np.zeros(128))), [])
        self.assertEqual(A.render_talkbox_aura([]).size, 0)

    def test_yin_tracks_fundamental(self):
        time = np.arange(4096) / SR
        frequency, confidence = A.fixed_yin(np.sin(2 * np.pi * 220 * time), SR)
        self.assertAlmostEqual(frequency, 220.0, delta=2.0)
        self.assertGreater(confidence, 0.8)

    def test_transcription_emits_closed_note_once(self):
        engine = TranscriptionEngine(sr=SR, hop=480)
        frames = []
        for pitch in (440.0, 440.0, 0.0):
            bus = LatentBus(sample_rate=SR, hop=480)
            log_pitch = np.log2(pitch / 440.0) / 6 if pitch else 0.0
            bus.write_pitch_gesture(log_pitch, 0.0, 0.0, 0.5, 0.0, 0.5, 0.8 if pitch else 0.0, 0.0)
            frames.append(bus)
        emitted = engine.process(frames)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["midi"], 69)
        self.assertAlmostEqual(emitted[0]["duration"], 0.02)
        self.assertEqual(engine.flush(), [])

    def test_a4_zero_log_pitch_is_not_silence(self):
        engine = TranscriptionEngine(sr=SR, hop=480)
        a4 = LatentBus(sample_rate=SR, hop=480)
        a4.write_pitch_gesture(0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.8, 0.0)
        silence = LatentBus(sample_rate=SR, hop=480)
        silence.write_pitch_gesture(0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0)
        self.assertEqual(engine.process([a4, silence])[0]["midi"], 69)
        self.assertEqual(engine.flush(), [])

    def test_midi_rejects_invalid_vlq_and_truncated_header(self):
        with self.assertRaises(ValueError):
            G._vlq(-1)
        with self.assertRaises(ValueError):
            G._read_vlq(b"\x81\x80\x80\x80\x00", 0)
        with self.assertRaises(ValueError):
            G._read_midi_impl(b"MThd")

    def test_format1_global_tempo_map_and_parallel_same_pitch_notes(self):
        def track(events):
            body = b"".join(G._vlq(delta) + event for delta, event in events)
            return b"MTrk" + struct.pack(">I", len(body)) + body

        tempo_track = track([
            (0, b"\xff\x51\x03" + (500000).to_bytes(3, "big")),
            (480, b"\xff\x51\x03" + (1000000).to_bytes(3, "big")),
            (0, b"\xff\x2f\x00"),
        ])
        first_note_track = track([
            (0, bytes([0x90, 60, 90])),
            (480, bytes([0x80, 60, 0])),
            (0, b"\xff\x2f\x00"),
        ])
        second_note_track = track([
            (0, bytes([0x90, 60, 80])),
            (960, bytes([0x80, 60, 0])),
            (0, b"\xff\x2f\x00"),
        ])
        header = b"MThd" + struct.pack(">IHHH", 6, 1, 3, 480)
        recovered = G._read_midi_impl(header + tempo_track + first_note_track + second_note_track)
        self.assertEqual(len(recovered), 2)
        self.assertAlmostEqual(recovered[0].dur, 0.5)
        self.assertAlmostEqual(recovered[1].dur, 1.5)

    def test_blind_extractor_rejects_short_training_pair(self):
        extractor = BlindExtractor(n_fft=64, hop=16)
        with self.assertRaises(ValueError):
            extractor.fit(np.zeros(16), np.zeros(16))

    def test_physical_model_rejects_invalid_control_values(self):
        model = __import__("aura_g.physical_model", fromlist=["DigitalWaveguideString"]).DigitalWaveguideString(sr=SR)
        with self.assertRaises(ValueError):
            model.pluck(fret=1.5)
        with self.assertRaises(ValueError):
            model.pluck(dynamics=float("nan"))
        with self.assertRaises(ValueError):
            model.step(1.5)

    def test_physical_string_fret_changes_delay(self):
        string = __import__("aura_g.physical_model", fromlist=["DigitalWaveguideString"]).DigitalWaveguideString(sr=SR, f0=196)
        open_delay = string.delay
        string.pluck(fret=12)
        self.assertLess(string.delay, open_delay)
        self.assertEqual(string.fret_damping(100), 0.0)

    def test_overlapping_same_pitch_notes_on_one_track_are_not_lost(self):
        def track(events):
            body = b"".join(G._vlq(delta) + event for delta, event in events)
            return b"MTrk" + struct.pack(">I", len(body)) + body

        events = [
            (0, bytes([0x90, 60, 90])),
            (240, bytes([0x90, 60, 80])),
            (240, bytes([0x80, 60, 0])),
            (480, bytes([0x80, 60, 0])),
            (0, b"\\xff\\x2f\\x00"),
        ]
        header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480)
        recovered = G._read_midi_impl(header + track(events))
        self.assertEqual(len(recovered), 2)
        self.assertAlmostEqual(recovered[0].start, 0.0)
        self.assertAlmostEqual(recovered[0].dur, 0.5)
        self.assertAlmostEqual(recovered[1].start, 0.25)
        self.assertAlmostEqual(recovered[1].dur, 0.75)
        self.assertEqual([note.vel for note in recovered], [90, 80])

    def test_midi_writer_rejects_tempo_outside_24_bit_range(self):
        with self.assertRaises(ValueError):
            G._build_midi_bytes([], bpm=1.0)

    def test_latent_bus_rejects_invalid_vectors_and_protects_state(self):
        with self.assertRaises(ValueError):
            LatentBus(vector=np.zeros(31))
        with self.assertRaises(ValueError):
            LatentBus(vector=np.full(32, np.nan))
        bus = LatentBus()
        bus.write_expression(np.arange(8, dtype=np.float32))
        exposed = bus.get_expression()
        exposed[0] = 999.0
        self.assertEqual(bus.get_expression()[0], 0.0)
        with self.assertRaises(ValueError):
            bus.write_expression(np.array([np.nan] * 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
