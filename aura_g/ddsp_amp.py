"""Differentiable guitar amplifier blocks with peaking, high-pass, and low-pass biquads."""
import numpy as np
import torch
import torch.nn as nn


class Biquad(nn.Module):
    """Differentiable frequency-domain RBJ biquad filter for complete audio blocks."""
    FILTER_TYPES = {"peaking", "highpass", "lowpass"}

    def __init__(self, sr=48000, f0=500.0, Q=1.0, gain_db=0.0, filter_type="peaking"):
        super().__init__()
        if not isinstance(sr, int) or sr <= 0:
            raise ValueError("sample rate must be a positive integer")
        if (not np.all(np.isfinite([f0, Q, gain_db])) or f0 <= 0 or Q <= 0 or
                filter_type not in self.FILTER_TYPES):
            raise ValueError("frequency and Q must be finite and positive; gain finite; filter type supported")
        self.sr = sr
        self.filter_type = filter_type
        self.f0 = nn.Parameter(torch.tensor(float(f0)))
        self.Q = nn.Parameter(torch.tensor(float(Q)))
        self.gain_db = nn.Parameter(torch.tensor(float(gain_db)))

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim < 1 or not x.is_floating_point():
            raise ValueError("biquad input must be a floating-point tensor with a time dimension")
        if x.shape[-1] == 0:
            return x
        if not torch.isfinite(x).all():
            raise ValueError("biquad input contains NaN or infinity")
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError("biquad supports float32 and float64 audio tensors")

        f0 = torch.nan_to_num(self.f0, nan=500.0, posinf=self.sr * 0.49,
                              neginf=1.0).to(device=x.device, dtype=x.dtype).clamp(1.0, self.sr * 0.49)
        q = torch.nan_to_num(self.Q, nan=1.0, posinf=50.0,
                             neginf=0.05).to(device=x.device, dtype=x.dtype).clamp(0.05, 50.0)
        gain_db = torch.nan_to_num(self.gain_db, nan=0.0, posinf=36.0,
                                   neginf=-36.0).to(device=x.device, dtype=x.dtype).clamp(-36.0, 36.0)
        w0 = 2.0 * np.pi * f0 / self.sr
        cosine, sine = torch.cos(w0), torch.sin(w0)
        alpha = sine / (2.0 * q)

        if self.filter_type == "lowpass":
            b0 = (1.0 - cosine) / 2.0
            b1 = 1.0 - cosine
            b2 = (1.0 - cosine) / 2.0
            a0, a1, a2 = 1.0 + alpha, -2.0 * cosine, 1.0 - alpha
        elif self.filter_type == "highpass":
            b0 = (1.0 + cosine) / 2.0
            b1 = -(1.0 + cosine)
            b2 = (1.0 + cosine) / 2.0
            a0, a1, a2 = 1.0 + alpha, -2.0 * cosine, 1.0 - alpha
        else:
            amplitude = torch.pow(10.0, gain_db / 40.0)
            b0, b1, b2 = 1.0 + alpha * amplitude, -2.0 * cosine, 1.0 - alpha * amplitude
            a0, a1, a2 = 1.0 + alpha / amplitude, -2.0 * cosine, 1.0 - alpha / amplitude

        spectrum = torch.fft.rfft(x, dim=-1)
        frequencies = torch.fft.rfftfreq(x.shape[-1], 1.0 / self.sr).to(device=x.device, dtype=x.dtype)
        z = torch.exp(-1j * 2.0 * np.pi * frequencies / self.sr)
        numerator = (b0 / a0) + (b1 / a0) * z + (b2 / a0) * z.square()
        denominator = 1.0 + (a1 / a0) * z + (a2 / a0) * z.square()
        response = numerator / denominator
        return torch.fft.irfft(spectrum * response, n=x.shape[-1], dim=-1)


class Preamp(nn.Module):
    def __init__(self, sr=48000):
        super().__init__()
        self.drive = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.eq_bass = Biquad(sr, f0=100, Q=0.7, gain_db=0.0)
        self.eq_mid = Biquad(sr, f0=800, Q=1.0, gain_db=0.0)
        self.eq_treble = Biquad(sr, f0=3000, Q=0.7, gain_db=0.0)

    def forward(self, x, drive_scale=1.0):
        drive = torch.nan_to_num(self.drive, nan=1.0).clamp(0.0, 20.0)
        bias = torch.nan_to_num(self.bias, nan=0.0).clamp(-2.0, 2.0)
        x = torch.tanh(drive * drive_scale * x + bias)
        return self.eq_treble(self.eq_mid(self.eq_bass(x)))


class PowerAmp(nn.Module):
    def __init__(self, sr=48000, envelope_seconds=0.02):
        super().__init__()
        if not isinstance(sr, int) or sr <= 0 or not np.isfinite(envelope_seconds) or envelope_seconds <= 0:
            raise ValueError("sample rate and envelope time must be finite and positive")
        self.sr = sr
        self.alpha = float(np.exp(-1.0 / (envelope_seconds * sr)))
        self.sag = nn.Parameter(torch.tensor(0.3))
        self.saturation = nn.Parameter(torch.tensor(0.7))

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim < 1 or not x.is_floating_point():
            raise ValueError("power amp input must be a floating-point tensor with a time dimension")
        if x.shape[-1] == 0:
            return x
        if not torch.isfinite(x).all():
            raise ValueError("power amp input contains NaN or infinity")
        env = torch.abs(x)
        env_smooth = torch.zeros_like(env)
        acc = torch.zeros_like(env[..., 0])
        for index in range(env.shape[-1]):
            acc = self.alpha * acc + (1.0 - self.alpha) * env[..., index]
            env_smooth[..., index] = acc
        sag = torch.nan_to_num(self.sag, nan=0.3).clamp(0.0, 10.0)
        saturation = torch.nan_to_num(self.saturation, nan=0.7).clamp(0.0, 10.0)
        gain_reduction = 1.0 / (1.0 + sag * env_smooth)
        return torch.tanh(saturation * x * gain_reduction)


class OutputTransformer(nn.Module):
    def __init__(self, sr=48000):
        super().__init__()
        self.hp = Biquad(sr, f0=80, Q=0.707, filter_type="highpass")
        self.lp = Biquad(sr, f0=min(6000, sr * 0.45), Q=0.707, filter_type="lowpass")

    def forward(self, x):
        return self.lp(self.hp(x))


class DDSPAmp(nn.Module):
    """Interpretable DDSP amp: preamp -> power amp -> output high/low-pass."""
    def __init__(self, sr=48000):
        super().__init__()
        if not isinstance(sr, int) or sr <= 0:
            raise ValueError("sample rate must be a positive integer")
        self.sr = sr
        self.preamp = Preamp(sr)
        self.power = PowerAmp(sr)
        self.transformer = OutputTransformer(sr)

    def forward(self, x, latent: torch.Tensor = None):
        if not isinstance(x, torch.Tensor) or x.ndim < 1 or not x.is_floating_point():
            raise ValueError("DDSP input must be a floating-point tensor with a time dimension")
        if latent is None:
            drive_scale = 1.0
        else:
            if not isinstance(latent, torch.Tensor) or latent.shape[-1] != 32:
                raise ValueError("latent control must be a tensor with final dimension 32")
            if not torch.isfinite(latent).all():
                raise ValueError("latent control contains NaN or infinity")
            drive_scale = (0.5 + latent[..., 7].mean()).clamp(0.0, 2.0)
        x = self.preamp(x, drive_scale=drive_scale)
        return self.transformer(self.power(x))
