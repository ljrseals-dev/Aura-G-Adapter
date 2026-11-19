"""Integration smoke tests for the Torch/torchaudio AURA-G processing path."""
import unittest
import numpy as np

from aura_g.pipeline import AURAG


class PipelineSmokeTests(unittest.TestCase):
    def test_empty_and_short_audio_are_safe(self):
        engine = AURAG(sr=48000, hop=480)
        empty, events, buses = engine.process(np.zeros(0, dtype=np.float32))
        self.assertEqual(empty.size, 0)
        self.assertEqual(events, [])
        self.assertEqual(buses, [])

        short = np.linspace(-0.1, 0.1, 128, dtype=np.float32)
        output, events, buses = engine.process(short)
        np.testing.assert_array_equal(output, short)
        self.assertEqual(events, [])
        self.assertEqual(buses, [])

    def test_full_path_preserves_shape_and_finite_output(self):
        engine = AURAG(sr=48000, hop=480)
        time = np.arange(4096, dtype=np.float32) / 48000
        audio = (0.15 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
        output, events, buses = engine.process(audio)
        self.assertEqual(output.shape, audio.shape)
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertGreater(len(buses), 0)
        self.assertTrue(all(np.all(np.isfinite(bus.read())) for bus in buses))
        final_events = engine.flush_transcription()
        self.assertIsInstance(final_events, list)

    def test_rejects_non_finite_and_stereo_input(self):
        engine = AURAG(sr=48000, hop=480)
        with self.assertRaises(ValueError):
            engine.process(np.array([0.0, np.nan], dtype=np.float32))
        with self.assertRaises(ValueError):
            engine.process(np.zeros((2, 128), dtype=np.float32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
