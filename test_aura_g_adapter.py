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
import mouth_tract as M

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
        unvoiced = LatentBus(sample_rate=SR, hop=480)
        unvoiced.write_pitch_gesture(0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.8, 0.0)
        unvoiced.write_pitch_confidence(0.0)
        self.assertEqual(TranscriptionEngine(sr=SR, hop=480).process([unvoiced]), [])

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
            (0, bytes([0xFF, 0x2F, 0x00])),
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
        with self.assertRaises(ValueError):
            G._build_midi_bytes([], bpm=1e-320)
        with self.assertRaises(ValueError):
            G._build_midi_bytes([], bpm="fast")

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

    def test_waveguide_rejects_above_nyquist_fundamentals(self):
        from aura_g.physical_model import DigitalWaveguideString, BodyResonance
        with self.assertRaises(ValueError):
            DigitalWaveguideString(sr=SR, f0=SR)
        string = DigitalWaveguideString(sr=SR, f0=1000)
        with self.assertRaises(ValueError):
            string.pluck(fret=60)
        with self.assertRaises(ValueError):
            BodyResonance(sr=SR, modes=[[440.0, 1.0, float("nan")]])

    def test_tuned_string_rejects_non_finite_controls(self):
        string = A.TunedWaveguideString(SR)
        with self.assertRaises(ValueError):
            string.render(440.0, 0.1, dynamics=float("nan"))
        with self.assertRaises(ValueError):
            string.render(440.0, 0.1, pick_lowpass=float("nan"))
        with self.assertRaises(ValueError):
            A.aura_g_transcribe(np.zeros(1024), SR, hop=0.5)

    def test_midi_bridge_fallback_handles_empty_and_exact_window(self):
        self.assertEqual(B.transcribe(np.zeros(0), SR), [])
        time = np.arange(int(0.046 * SR)) / SR
        audio = 0.2 * np.sin(2 * np.pi * 440 * time)
        notes = B.transcribe(audio, SR, min_note=0.0)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].midi, 69)

    def test_midi_bridge_rejects_invalid_audio(self):
        with self.assertRaises(ValueError):
            B.transcribe(np.zeros((2, 1024)), SR)
        with self.assertRaises(ValueError):
            B.transcribe(np.zeros(1024), 0)

    def test_yin_silence_is_not_misclassified_as_a_pitch(self):
        self.assertEqual(A.fixed_yin(np.zeros(4096), SR), (0.0, 0.0))
        self.assertEqual(B._yin(np.zeros(2208), SR), (0.0, 1.0))

    def test_unplucked_physical_string_starts_silent(self):
        from aura_g.physical_model import DigitalWaveguideString
        string = DigitalWaveguideString(sr=SR, f0=196.0)
        np.testing.assert_array_equal(string.step(512), np.zeros(512))

    def test_front_end_rejects_fractional_sample_clock_parameters(self):
        with self.assertRaises(ValueError):
            FrontEndAnalyzer(sr=48000.5)
        with self.assertRaises(ValueError):
            FrontEndAnalyzer(hop=480.5)
        with self.assertRaises(ValueError):
            A.fixed_yin(np.ones(4096), SR, threshold=1.5)

    def test_mixer_rejects_fractional_output_length(self):
        from aura_g.mixer import RoutingMatrix
        mixer = RoutingMatrix()
        with self.assertRaises(ValueError):
            mixer.blend(np.zeros(4), np.zeros(4), np.zeros(4), 3.5)

    def test_midi_meta_lengths_use_variable_length_quantities(self):
        text_meta = bytes([0xFF, 0x01]) + G._vlq(128) + (b"A" * 128)
        events = [
            (0, text_meta),
            (0, bytes([0x90, 60, 90])),
            (480, bytes([0x80, 60, 0])),
            (0, bytes([0xFF, 0x2F, 0x00])),
        ]
        body = b"".join(G._vlq(delta) + event for delta, event in events)
        track = b"MTrk" + struct.pack(">I", len(body)) + body
        header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480)
        recovered = G._read_midi_impl(header + track)
        self.assertEqual(len(recovered), 1)
        self.assertAlmostEqual(recovered[0].dur, 0.5)

    def test_midi_rejects_data_after_end_of_track(self):
        events = [(0, bytes([0xFF, 0x2F, 0x00])), (0, bytes([0x90, 60, 90]))]
        body = b"".join(G._vlq(delta) + event for delta, event in events)
        track = b"MTrk" + struct.pack(">I", len(body)) + body
        header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480)
        with self.assertRaises(ValueError):
            G._read_midi_impl(header + track)

    def test_midi_writer_accepts_numeric_string_tempo_without_type_error(self):
        data = G._build_midi_bytes([B.Note(0.0, 0.5, 60, 90)], bpm="120")
        recovered = G._read_midi_impl(data)
        self.assertEqual(len(recovered), 1)
        self.assertAlmostEqual(recovered[0].dur, 0.5)

    def test_latent_bus_vector_property_is_copy_only(self):
        bus = LatentBus()
        exposed = bus.vector
        exposed[0] = 123.0
        self.assertEqual(bus.read()[0], 0.0)

    def test_pitch_synthesis_rejects_above_nyquist(self):
        with self.assertRaises(ValueError):
            G.pluck(127, 0.1, 0.8, 8000)

    def test_transcriber_rejects_fractional_sample_clocks_and_non_finite_gate(self):
        with self.assertRaises(ValueError):
            TranscriptionEngine(sr=48000, hop=480.5)
        with self.assertRaises(ValueError):
            TranscriptionEngine(sr=48000, gate=float("nan"))

    def test_yin_validates_parameters_even_for_silent_frames(self):
        with self.assertRaises(ValueError):
            A.fixed_yin(np.zeros(128), 0)
        with self.assertRaises(ValueError):
            B._yin(np.zeros(128), SR, thr=1.5)

    def test_transcription_hop_duration_survives_sample_rate_conversion(self):
        from unittest.mock import patch
        with patch("aura_g_adapter.FrontEndAnalyzer") as analyzer:
            analyzer.return_value.process.return_value = []
            A.aura_g_transcribe(np.zeros(44100), 44100, hop=441)
            analyzer.assert_called_once_with(sr=48000, hop=480)

    def test_blind_extractor_short_apply_bypasses_without_silencing(self):
        extractor = BlindExtractor(n_fft=64, hop=16)
        training = np.random.default_rng(7).normal(size=256)
        extractor.fit(training, training * 0.5)
        short = np.linspace(-0.2, 0.2, 16)
        np.testing.assert_array_equal(extractor.apply(short), short)

    def test_complex_audio_is_rejected_instead_of_discarded(self):
        with self.assertRaises(ValueError):
            A.aura_g_transcribe(np.array([1 + 1j, 0j]), SR)
        with self.assertRaises(ValueError):
            list(FrontEndAnalyzer(sr=SR).process(np.array([1 + 1j, 0j])))
        with self.assertRaises(ValueError):
            B.transcribe(np.array([1 + 1j, 0j]), SR)
        with self.assertRaises(ValueError):
            BlindExtractor(n_fft=4, hop=2)._stft(np.array([1 + 1j, 0j, 0j, 0j]))

    def test_latent_bus_rejects_complex_values(self):
        with self.assertRaises(ValueError):
            LatentBus(vector=np.ones(32, dtype=np.complex128))
        with self.assertRaises(ValueError):
            LatentBus().write_expression(np.ones(8, dtype=np.complex128))

    def test_mouth_tract_validates_shapes_controls_and_short_formants(self):
        areas = M.area_function(0.5)
        self.assertEqual(areas.shape, (48,))
        self.assertTrue(np.all(np.isfinite(areas)))
        with self.assertRaises(ValueError):
            M.area_function(float("nan"))
        with self.assertRaises(ValueError):
            M.Tract(rg=1.0)
        tract = M.Tract()
        with self.assertRaises(ValueError):
            tract.process(np.zeros(8), np.zeros(7))
        with self.assertRaises(ValueError):
            tract.process(np.zeros(8), np.zeros(8), block=0)
        with self.assertRaises(ValueError):
            tract.process(np.zeros(8, dtype=np.complex128), np.zeros(8))
        self.assertEqual(tract.process(np.zeros(0), np.zeros(0)).shape, (0,))
        self.assertIsInstance(M.formants(0.5, n=128), list)
        with self.assertRaises(ValueError):
            M.formants(0.5, n=2)

    def test_yin_pitch_bounds_must_stay_below_nyquist(self):
        with self.assertRaises(ValueError):
            A.fixed_yin(np.ones(4096), 2000, fmax=1200)
        with self.assertRaises(ValueError):
            B._yin(np.ones(4096), 2000, fmax=1200)
        with self.assertRaises(ValueError):
            FrontEndAnalyzer(sr=2000, fmax=1200)

    def test_registered_transcriber_contract_is_validated(self):
        with self.assertRaises(TypeError):
            B.register_transcriber(123)
        B.register_transcriber(lambda audio, sr: [B.Note(0.0, 0.1, 128, 90)])
        try:
            with self.assertRaises(ValueError):
                B.transcribe(np.zeros(1024), SR)
        finally:
            B.register_transcriber(None)

    def test_mixer_rejects_corrupted_weights(self):
        from aura_g.mixer import RoutingMatrix
        mixer = RoutingMatrix()
        mixer.weights[0] = float("nan")
        with self.assertRaises(ValueError):
            mixer.blend(np.zeros(4), np.zeros(4), np.zeros(4), 4)

    def test_zero_velocity_is_silence_and_not_a_midi_note_on(self):
        with self.assertRaises(ValueError):
            G._build_midi_bytes([B.Note(0.0, 0.2, 60, 0)])
        silent = A.render_talkbox_aura([B.Note(0.0, 0.2, 60, 0)], tail=0.0)
        np.testing.assert_array_equal(silent, np.zeros_like(silent))

    def test_physical_model_tracks_latent_pitch_and_does_not_repluck_each_block(self):
        from aura_g.physical_model import PhysicalGuitarModel, BodyResonance
        def bus_for(midi, onset, dynamics=0.8):
            bus = LatentBus(sample_rate=SR, hop=480)
            log_pitch = (midi - 69) / 72.0
            bus.write_pitch_gesture(log_pitch, onset, 0.0, 0.5, 0.0, 0.3,
                                    dynamics, 0.0)
            return bus
        model = PhysicalGuitarModel(sr=SR)
        calls = []
        original_pluck = model.string.pluck
        def tracked_pluck(*args, **kwargs):
            calls.append((args, kwargs))
            return original_pluck(*args, **kwargs)
        model.string.pluck = tracked_pluck
        output = model.synthesize_sequence([bus_for(69, 0.0), bus_for(72, 1.0)], 960)
        self.assertEqual(len(calls), 2)
        self.assertTrue(np.all(np.isfinite(output)))
        model.synthesize_sequence([bus_for(72, 0.0)], 480)
        self.assertEqual(len(calls), 2)
        model.synthesize_sequence([bus_for(72, 0.0, dynamics=0.0)], 480)
        model.synthesize_sequence([bus_for(72, 0.0)], 480)
        self.assertEqual(len(calls), 3)

        source = np.random.default_rng(9).normal(size=1024)
        whole = BodyResonance(sr=SR)(source)
        split_filter = BodyResonance(sr=SR)
        split = np.concatenate([split_filter(source[:400]), split_filter(source[400:])])
        np.testing.assert_allclose(split, whole, rtol=1e-12, atol=1e-12)

    def test_front_end_preserves_unvoiced_confidence_separately_from_a4(self):
        noise = np.random.default_rng(123).normal(0.0, 0.1, 2048)
        buses = list(FrontEndAnalyzer(sr=SR, hop=480).process(noise))
        self.assertTrue(buses)
        self.assertLess(buses[0].get_pitch_confidence(), 0.5)
        self.assertEqual(TranscriptionEngine(sr=SR, hop=480).process(buses), [])

    def test_mixer_morph_clamps_unbounded_finite_macros(self):
        from aura_g.mixer import RoutingMatrix
        mixer = RoutingMatrix()
        latent = np.zeros(32)
        latent[16:19] = [1e300, -1e300, 1e300]
        weights = mixer.morph(latent)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights >= 0))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_physical_model_resets_pitch_after_invalid_control_frame(self):
        from aura_g.physical_model import PhysicalGuitarModel
        model = PhysicalGuitarModel(sr=SR)
        calls = []
        original = model.string.pluck
        def tracked(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        model.string.pluck = tracked
        valid = LatentBus(sample_rate=SR, hop=480)
        valid.write_pitch_gesture((72 - 69) / 72.0, 0.0, 0.0, 0.5, 0.0, 0.3, 0.8, 0.0)
        invalid = LatentBus(sample_rate=SR, hop=480)
        invalid.write_pitch_gesture(10.0, 0.0, 0.0, 0.5, 0.0, 0.3, 0.8, 0.0)
        model.synthesize_sequence([valid], 480)
        model.synthesize_sequence([invalid], 480)
        model.synthesize_sequence([valid], 480)
        self.assertEqual(len(calls), 2)

    def test_blind_extractor_overlap_add_covers_both_clip_edges(self):
        extractor = BlindExtractor(n_fft=64, hop=16)
        training = np.random.default_rng(8).normal(size=1024)
        extractor.fit(training, training * 0.5)
        output = extractor.apply(np.full(1000, 0.2))
        self.assertEqual(output.shape, (1000,))
        self.assertGreater(output[0], 0.05)
        self.assertGreater(output[-1], 0.05)
        with self.assertRaises(ValueError):
            BlindExtractor(n_fft=64, hop=40)
        with self.assertRaises(ValueError):
            extractor.apply(np.array([1 + 1j, 0j]))

    def test_blind_extractor_requires_usable_window_size(self):
        with self.assertRaises(ValueError):
            BlindExtractor(n_fft=2, hop=1)

    def test_demo_wav_decoder_centers_unsigned_pcm_and_rejects_non_finite(self):
        from demo_aura_g import decode_wav_audio
        decoded = decode_wav_audio(np.array([0, 128, 255], dtype=np.uint8))
        self.assertAlmostEqual(decoded[0], -1.0)
        self.assertAlmostEqual(decoded[1], 0.0)
        self.assertGreater(decoded[2], 0.98)
        stereo = decode_wav_audio(np.array([[0, 255], [128, 128]], dtype=np.uint8))
        self.assertEqual(stereo.shape, (2,))
        with self.assertRaises(ValueError):
            decode_wav_audio(np.array([0.0, np.nan]))
        self.assertEqual(decode_wav_audio(np.zeros(0, dtype=np.uint8)).shape, (0,))

    def test_front_end_yin_rejects_complex_frames(self):
        from aura_g.front_end import yin_pitch
        with self.assertRaises(ValueError):
            yin_pitch(np.array([1 + 1j] * 128), SR)

    def test_mixer_rejects_complex_weight_state(self):
        from aura_g.mixer import RoutingMatrix
        mixer = RoutingMatrix()
        mixer.weights = np.array([0.5 + 1j, 0.3, 0.2])
        with self.assertRaises(ValueError):
            mixer.blend(np.zeros(4), np.zeros(4), np.zeros(4), 4)


    def test_transcription_preserves_midi_zero_and_measured_confidence(self):
        engine = TranscriptionEngine(sr=SR, hop=480)
        bus = LatentBus(sample_rate=SR, hop=480)
        bus.write_pitch_gesture((0 - 69) / 72.0, 0.0, 0.0, 0.5, 0.0, 0.3, 0.8, 0.0)
        bus.write_pitch_confidence(0.73)
        engine.process([bus])
        event = engine.flush()[0]
        self.assertEqual(event["midi"], 0)
        self.assertAlmostEqual(event["confidence"], 0.73, places=6)
        self.assertNotIn("_confidence_sum", event)

    def test_talkbox_renderers_accept_generators_and_empty_iterables(self):
        self.assertEqual(A.render_talkbox_aura(iter(()), tail=0.0).size, 0)
        self.assertEqual(B.render_talkbox(iter(()), tail=0.0).size, 0)
        self.assertEqual(B.render_talkbox([B.Note(0.0, 0.0, 60, 90)], tail=0.0).size, 0)
        note = B.Note(0.0, 0.05, 69, 90)
        rendered_aura = A.render_talkbox_aura(iter([note]), tail=0.0)
        rendered_bridge = B.render_talkbox(iter([note]), tail=0.0)
        self.assertGreater(rendered_aura.size, 0)
        self.assertGreater(rendered_bridge.size, 0)
        self.assertTrue(np.all(np.isfinite(rendered_aura)))
        self.assertTrue(np.all(np.isfinite(rendered_bridge)))

    def test_physical_sequence_bypasses_midi_pitch_above_sample_rate_nyquist(self):
        from aura_g.physical_model import PhysicalGuitarModel
        model = PhysicalGuitarModel(sr=22050)
        bus = LatentBus(sample_rate=22050, hop=220)
        bus.write_pitch_gesture((127 - 69) / 72.0, 1.0, 0.0, 0.5, 0.0, 0.3, 0.8, 0.0)
        bus.write_pitch_confidence(1.0)
        output = model.synthesize_sequence([bus], 220)
        self.assertEqual(output.shape, (220,))
        self.assertTrue(np.all(np.isfinite(output)))
        model.string.f0 = 12543.0
        with self.assertRaises(ValueError):
            model.string.pluck(fret=0, pluck_pos=0.3, dynamics=0.8)

if __name__ == "__main__":
    unittest.main(verbosity=2)
