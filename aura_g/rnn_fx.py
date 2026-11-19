"""Sample-rate-independent RNN FX with interpolation-filter resampling."""
import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as F


class InterpolationFilter(nn.Module):
    """FIR prefilter followed by rational resampling when a rate change is needed."""
    def __init__(self, taps=64, ratio=1.0, cutoff=0.45):
        super().__init__()
        if not isinstance(taps, int) or taps < 2:
            raise ValueError("taps must be an integer >= 2")
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("resampling ratio must be finite and positive")
        if not np.isfinite(cutoff) or not 0 < cutoff < 0.5:
            raise ValueError("cutoff must be between 0 and Nyquist")
        self.taps = taps
        n = np.arange(taps) - (taps - 1) / 2
        h = np.sinc(2 * cutoff * n) * np.kaiser(taps, 8.0)
        h = h / h.sum()
        self.register_buffer("kernel", torch.tensor(h, dtype=torch.float32))
        self.ratio = float(ratio)

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError("resampler input must have shape [batch, samples]")
        if not x.is_floating_point():
            raise TypeError("resampler input must be floating point")
        if x.shape[-1] == 0:
            return x
        # A same-rate path must be transparent: filtering here added two
        # unnecessary FIR stages and 63 samples of group delay per stage.
        if self.ratio == 1.0:
            return x
        kernel = self.kernel.to(device=x.device, dtype=x.dtype).flip(0).view(1, 1, -1)
        padded = torch.nn.functional.pad(x.unsqueeze(1), (self.taps - 1, 0))
        filtered = torch.nn.functional.conv1d(padded, kernel).squeeze(1)
        new_length = max(1, int(round(filtered.shape[-1] * self.ratio)))
        return F.resample(filtered, filtered.shape[-1], new_length)


class GRUFX(nn.Module):
    """GRU-based nonlinear FX (chorus/flanger/delay approximation)."""
    def __init__(self, hidden=32, sr=48000):
        super().__init__()
        if not isinstance(hidden, int) or hidden <= 0:
            raise ValueError("hidden size must be a positive integer")
        if not isinstance(sr, int) or sr <= 0:
            raise ValueError("sample rate must be a positive integer")
        self.sr = sr
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)
        self.latent_proj = nn.Linear(32, hidden)

    def forward(self, x, latent: torch.Tensor = None):
        if x.ndim != 2 or not x.is_floating_point():
            raise ValueError("GRU input must be a floating [batch, samples] tensor")
        if x.shape[-1] == 0:
            return x
        if x.dtype != self.gru.weight_ih_l0.dtype:
            raise TypeError("GRU input dtype must match model parameter dtype")
        batch = x.shape[0]
        h0 = x.new_zeros(1, batch, self.gru.hidden_size)
        if latent is not None:
            if latent.shape != (batch, 32):
                raise ValueError("latent input must have shape [batch, 32]")
            if latent.device != x.device:
                latent = latent.to(device=x.device)
            if latent.dtype != x.dtype:
                latent = latent.to(dtype=x.dtype)
            h0 = torch.tanh(self.latent_proj(latent)).unsqueeze(0)
        y, _ = self.gru(x.unsqueeze(-1), h0)
        return self.out(y).squeeze(-1)


class SampleRateIndependentFX(nn.Module):
    """Wraps a GRU FX with interpolation filters on input and output."""
    def __init__(self, sr_in=48000, sr_internal=48000, hidden=32):
        super().__init__()
        if (not isinstance(sr_in, int) or sr_in <= 0 or
                not isinstance(sr_internal, int) or sr_internal <= 0):
            raise ValueError("sample rates must be positive integers")
        self.sr_in = sr_in
        self.sr_internal = sr_internal
        ratio_in = sr_internal / sr_in
        ratio_out = sr_in / sr_internal
        self.pre = InterpolationFilter(ratio=ratio_in)
        self.fx = GRUFX(hidden=hidden, sr=sr_internal)
        self.post = InterpolationFilter(ratio=ratio_out)

    def forward(self, x, latent=None):
        if x.ndim != 2:
            raise ValueError("FX input must have shape [batch, samples]")
        if x.shape[-1] == 0:
            return x
        original_length = x.shape[-1]
        x = self.pre(x)
        x = self.fx(x, latent)
        x = self.post(x)
        # Rational rounding at the two conversion boundaries can differ by
        # one sample. Keep the public contract duration/shape preserving.
        if x.shape[-1] > original_length:
            return x[..., :original_length]
        if x.shape[-1] < original_length:
            return torch.nn.functional.pad(x, (0, original_length - x.shape[-1]))
        return x
