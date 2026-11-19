"""Sample-rate-independent RNN FX with band-limited rational resampling."""
from fractions import Fraction
import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as F


class InterpolationFilter(nn.Module):
    """Band-limited polyphase resampler; a 1:1 rate path is transparent."""
    def __init__(self, ratio=1.0):
        super().__init__()
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("resampling ratio must be finite and positive")
        self.ratio = float(ratio)
        fraction = Fraction(self.ratio).limit_denominator(1000)
        self.up, self.down = fraction.numerator, fraction.denominator

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError("resampler input must have shape [batch, samples]")
        if not x.is_floating_point():
            raise TypeError("resampler input must be floating point")
        if x.shape[-1] == 0 or self.up == self.down:
            return x
        return F.resample(x, self.down, self.up)


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
    """Wraps a GRU FX with band-limited input and output rate conversion."""
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
        # The two rational conversions can round by one sample at the
        # boundaries. Preserve the public shape and duration contract.
        if x.shape[-1] > original_length:
            return x[..., :original_length]
        if x.shape[-1] < original_length:
            return torch.nn.functional.pad(x, (0, original_length - x.shape[-1]))
        return x
