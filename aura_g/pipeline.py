"""Full AURA-G pipeline: analyzer -> latent bus -> modules -> output + transcription."""
import numpy as np
import torch
from .front_end import FrontEndAnalyzer
from .ddsp_amp import DDSPAmp
from .rnn_fx import SampleRateIndependentFX
from .physical_model import PhysicalGuitarModel
from .transcription import TranscriptionEngine
from .mixer import RoutingMatrix


class AURAG:
    def __init__(self, sr=48000, hop=480):
        if (not isinstance(sr, (int, np.integer)) or sr <= 0 or
                not isinstance(hop, (int, np.integer)) or hop <= 0):
            raise ValueError("sample rate and hop must be positive integers")
        self.sr, self.hop = int(sr), int(hop)
        self.analyzer = FrontEndAnalyzer(sr=self.sr, hop=self.hop)
        self.amp = DDSPAmp(sr=self.sr).eval()
        self.fx = SampleRateIndependentFX(sr_in=self.sr, sr_internal=self.sr).eval()
        self.phys = PhysicalGuitarModel(sr=self.sr)
        self.transcriber = TranscriptionEngine(sr=self.sr, hop=self.hop)
        self.mixer = RoutingMatrix()

    def process(self, audio: np.ndarray, use_ddsp=True, use_fx=True, use_phys=True):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("audio must be mono and one-dimensional")
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio contains NaN or infinite samples")
        if len(audio) == 0:
            return audio.copy(), [], []
        bus_history = list(self.analyzer.process(audio))
        if not bus_history:
            # Preserve the input for clips shorter than the analysis window.
            return audio.copy(), [], []
        x = torch.from_numpy(audio.copy()).unsqueeze(0)
        with torch.no_grad():
            mean_latent = torch.as_tensor(
                np.mean([b.read() for b in bus_history], axis=0),
                dtype=torch.float32, device=x.device).unsqueeze(0)
            amp_out = self.amp(x, mean_latent).squeeze(0).cpu().numpy() if use_ddsp else np.zeros_like(audio)
            fx_out = self.fx(x, mean_latent).squeeze(0).cpu().numpy() if use_fx else np.zeros_like(audio)

        if use_phys:
            pg = bus_history[0].get_pitch_gesture()
            # Zero log-pitch is A4; use neutral tuning only while dynamics is gated.
            f0 = 440.0 * (2 ** (float(pg[0]) * 6.0)) if pg[6] >= self.transcriber.gate else 196.0
            self.phys.string.f0 = f0 if f0 > 0 else 196.0
            phys_out = self.phys.synthesize(len(audio), fret=0, pluck_pos=0.3,
                                            dynamics=float(np.clip(pg[6], 0, 1)))
        else:
            phys_out = np.zeros_like(audio)

        self.mixer.morph(bus_history[0].read())
        out = self.mixer.blend(amp_out, fx_out, phys_out, len(audio))
        events = self.transcriber.process(bus_history)
        return out.astype(np.float32), events, bus_history

    def flush_transcription(self):
        """Return the final note event after the caller signals end-of-stream."""
        return self.transcriber.flush()
