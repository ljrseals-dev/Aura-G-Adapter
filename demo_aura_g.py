"""Guitar audio -> full AURA-V guitar chain.  python demo_aura_g.py [input.wav]   writes transcribed.mid and guitar_chain_demo.wav (44.1 kHz, 16-bit, -14 LUFS)."""
import sys, numpy as np
from scipy.io import wavfile
import guitar_chain as C, guitar_gates as G, midi_bridge as B
if len(sys.argv) > 1:
    sr, x = wavfile.read(sys.argv[1]); x = x.astype(np.float64); x = x.mean(1) if x.ndim > 1 else x; x /= np.max(np.abs(x)) + 1e-12
else:
    sr, x = G.SR, G.guitar_audio()
r = C.guitar_chain(x, sr); B.write_midi("transcribed.mid", r["notes"])
wavfile.write("guitar_chain_demo.wav", 44100, np.round(r["out"] * 32767).astype(np.int16))
print(f"{len(r['notes'])} notes transcribed; limiter gain reduction {r['limiter_gr_db']:.1f} dB; wrote transcribed.mid and guitar_chain_demo.wav")
