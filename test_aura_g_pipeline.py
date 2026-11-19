"""Integration smoke tests for the Torch/torchaudio AURA-G processing path."""
import unittest
import numpy as np
import torch

from aura_g.pipeline import AURAG
from aura_g.rnn_fx import InterpolationFilter, SampleRateIndependentFX
from aura_g.ddsp_amp import Biquad, DDSPAmp


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

    def test_same_rate_resampler_is_transparent_and_empty_safe(self):
        resampler = InterpolationFilter(ratio=1.0)
        audio = torch.randn(2, 257)
        self.assertIs(resampler(audio), audio)
        empty = torch.empty(1, 0)
        self.assertEqual(resampler(empty).shape, (1, 0))

    def test_rnn_fx_resampling_preserves_public_shape(self):
        engine = SampleRateIndependentFX(sr_in=44100, sr_internal=48000)
        audio = torch.randn(1, 1024)
        latent = torch.zeros(1, 32)
        output = engine(audio, latent)
        self.assertEqual(output.shape, audio.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(engine(torch.empty(1, 0)).shape, (1, 0))
        with self.assertRaises(ValueError):
            engine.fx(audio, torch.zeros(1, 31))

    def test_pipeline_rejects_fractional_sample_clock(self):
        with self.assertRaises(ValueError):
            AURAG(sr=48000.5)
        with self.assertRaises(ValueError):
            AURAG(hop=480.5)

    def test_output_transformer_uses_real_highpass_and_lowpass_filters(self):
        highpass = Biquad(sr=48000, f0=200, Q=0.707, filter_type="highpass")
        lowpass = Biquad(sr=48000, f0=2000, Q=0.707, filter_type="lowpass")
        dc = torch.ones(1, 4096)
        hp_dc = highpass(dc)
        self.assertLess(float(hp_dc.abs().max()), 1e-5)
        time = torch.arange(4096, dtype=torch.float32) / 48000
        high_tone = torch.sin(2 * torch.pi * 10000 * time).unsqueeze(0)
        lp_high = lowpass(high_tone)
        self.assertLess(float(lp_high.square().mean().sqrt()), 0.1)

    def test_ddsp_validates_latent_and_keeps_parameters_differentiable(self):
        amp = DDSPAmp(sr=48000)
        audio = torch.randn(1, 2048, requires_grad=True)
        latent = torch.zeros(1, 32)
        output = amp(audio, latent)
        self.assertEqual(output.shape, audio.shape)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(amp.preamp.drive.grad)
        with self.assertRaises(ValueError):
            amp(audio.detach(), torch.zeros(1, 31))
        with self.assertRaises(ValueError):
            Biquad(filter_type="not-a-filter")


if __name__ == "__main__":
    unittest.main(verbosity=2)
