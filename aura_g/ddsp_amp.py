"""DDSP Guitar Amplifier — interpretable preamp/tone-stack/power-amp/transformer."""
import numpy as np
import torch
import torch.nn as nn


class Biquad(nn.Module):
    """Differentiable biquad peaking EQ (frequency-domain for speed)."""
    def __init__(self, sr=48000, f0=500.0, Q=1.0, gain_db=0.0):
        super().__init__()
        if sr <= 0:
            raise ValueError("sample rate must be positive")
        self.sr = sr
        self.f0 = nn.Parameter(torch.tensor(float(f0)))
        self.Q = nn.Parameter(torch.tensor(float(Q)))
        self.gain_db = nn.Parameter(torch.tensor(float(gain_db)))

    def forward(self, x):
        if x.shape[-1] == 0:
            return x
        f0 = self.f0.clamp(1.0, self.sr * 0.49)
        q = self.Q.clamp(0.05, 50.0)
        gain_db = self.gain_db.clamp(-36.0, 36.0)
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * f0 / self.sr
        alpha = torch.sin(w0) / (2 * q)
        b0, b1, b2 = 1 + alpha * A, -2 * torch.cos(w0), 1 - alpha * A
        a0, a1, a2 = 1 + alpha / A, -2 * torch.cos(w0), 1 - alpha / A
        X = torch.fft.rfft(x)
        w = torch.fft.rfftfreq(x.shape[-1], 1 / self.sr).to(x.device)
        z = torch.exp(-1j * 2 * np.pi * w / self.sr)
        H = (b0 / a0 + b1 / a0 * z + b2 / a0 * z ** 2) / (1 + a1 / a0 * z + a2 / a0 * z ** 2)
        return torch.fft.irfft(X * H, n=x.shape[-1])


class Preamp(nn.Module):
    def __init__(self, sr=48000):
        super().__init__()
        self.drive = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.eq_bass = Biquad(sr, f0=100, Q=0.7, gain_db=0.0)
        self.eq_mid = Biquad(sr, f0=800, Q=1.0, gain_db=0.0)
        self.eq_treble = Biquad(sr, f0=3000, Q=0.7, gain_db=0.0)

    def forward(self, x, drive_scale=1.0):
        x = torch.tanh(self.drive * drive_scale * x + self.bias)
        return self.eq_treble(self.eq_mid(self.eq_bass(x)))


class PowerAmp(nn.Module):
    def __init__(self):
        super().__init__()
        self.sag = nn.Parameter(torch.tensor(0.3))
        self.saturation = nn.Parameter(torch.tensor(0.7))

    def forward(self, x):
        if x.shape[-1] == 0:
            return x
        env = torch.abs(x)
        alpha = 0.999
        env_smooth = torch.zeros_like(env)
        acc = torch.zeros_like(env[..., 0])
        for i in range(env.shape[-1]):
            acc = alpha * acc + (1 - alpha) * env[..., i]
            env_smooth[..., i] = acc
        gain_reduction = 1.0 / (1.0 + self.sag.clamp(0.0, 10.0) * env_smooth)
        return torch.tanh(self.saturation.clamp(0.0, 10.0) * x * gain_reduction)


class OutputTransformer(nn.Module):
    def __init__(self, sr=48000):
        super().__init__()
        self.hp = Biquad(sr, f0=80, Q=0.7, gain_db=-3.0)
        self.lp = Biquad(sr, f0=6000, Q=0.7, gain_db=-3.0)

    def forward(self, x):
        return self.lp(self.hp(x))


class DDSPAmp(nn.Module):
    """Interpretable DDSP amp: preamp -> tone stack -> power amp -> transformer."""
    def __init__(self, sr=48000):
        super().__init__()
        self.preamp = Preamp(sr)
        self.power = PowerAmp()
        self.transformer = OutputTransformer(sr)

    def forward(self, x, latent: torch.Tensor = None):
        drive_scale = 1.0
        if latent is not None:
            drive_scale = (0.5 + latent[..., 7].mean()).clamp(0.0, 2.0)
        x = self.preamp(x, drive_scale=drive_scale)
        return self.transformer(self.power(x))
