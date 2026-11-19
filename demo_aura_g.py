"""Guitar audio -> AURA-G transcription -> MIDI file -> tuned waveguide string -> talk-box mouth.   python demo_aura_g.py [input.wav]"""
import sys, numpy as np
from scipy.io import wavfile
import aura_g_adapter as A, midi_bridge as B, harmonia_guitar as G
SR = 48000
if len(sys.argv) > 1:
    sr, x = wavfile.read(sys.argv[1]); x = x.astype(np.float64); x = x.mean(1) if x.ndim > 1 else x; x /= np.max(np.abs(x)) + 1e-12
else:
    ref = [B.Note(0.1 + 0.45 * i, 0.4, m, 90) for i, m in enumerate([52, 55, 59, 57, 55, 52, 50, 47, 64, 62])]; sr = SR
    x = np.zeros(int(SR * 5.0))
    for i, n in enumerate(ref): p = G.pluck(n.midi, n.dur, 0.8, SR, seed=i); a = int(n.start * SR); x[a:a + len(p)] += p[:len(x) - a]
    x /= np.max(np.abs(x))
notes = A.aura_g_transcribe(x, sr); B.write_midi("transcribed.mid", notes)
print(f"AURA-G transcribed {len(notes)} notes:", [(round(n.start, 2), n.midi, round(n.dur, 2)) for n in notes])
y = A.render_talkbox_aura(notes); wavfile.write("aura_g_talkbox_demo.wav", SR, (y * 32767).astype(np.int16))
print(f"wrote transcribed.mid and aura_g_talkbox_demo.wav ({len(y)/SR:.1f} s, peak {20*np.log10(np.max(np.abs(y))):.1f} dBFS)")
