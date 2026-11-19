"""Guitar audio -> AURA-G transcription -> MIDI -> tuned string -> talk-box renderer."""
import sys
import numpy as np
from scipy.io import wavfile
import aura_g_adapter as A
import midi_bridge as B
import harmonia_guitar as G

SR = 48000


def decode_wav_audio(samples):
    """Convert scipy WAV samples to finite, peak-normalized mono float64 audio."""
    raw = np.asarray(samples)
    if np.iscomplexobj(raw):
        raise ValueError("WAV audio must be real-valued")
    if np.issubdtype(raw.dtype, np.unsignedinteger):
        info = np.iinfo(raw.dtype)
        midpoint = (info.max + 1.0) / 2.0
        audio = (raw.astype(np.float64) - midpoint) / midpoint
    elif np.issubdtype(raw.dtype, np.signedinteger):
        info = np.iinfo(raw.dtype)
        scale = float(max(abs(info.min), info.max))
        audio = raw.astype(np.float64) / scale
    else:
        audio = np.asarray(raw, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not np.all(np.isfinite(audio)):
        raise ValueError("WAV input must be finite mono or multichannel audio")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0:
        audio = audio / peak
    return audio


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    if args:
        sr, samples = wavfile.read(args[0])
        x = decode_wav_audio(samples)
    else:
        reference = [B.Note(0.1 + 0.45 * i, 0.4, midi, 90)
                     for i, midi in enumerate([52, 55, 59, 57, 55, 52, 50, 47, 64, 62])]
        sr = SR
        x = np.zeros(int(SR * 5.0), dtype=np.float64)
        for index, note in enumerate(reference):
            plucked = G.pluck(note.midi, note.dur, 0.8, SR, seed=index)
            start = int(note.start * SR)
            x[start:start + len(plucked)] += plucked[:len(x) - start]
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 0.0:
            x /= peak

    notes = A.aura_g_transcribe(x, sr)
    B.write_midi("transcribed.mid", notes)
    print(f"AURA-G transcribed {len(notes)} notes:",
          [(round(note.start, 2), note.midi, round(note.dur, 2)) for note in notes])
    output = A.render_talkbox_aura(notes)
    wavfile.write("aura_g_talkbox_demo.wav", SR, (output * 32767).astype(np.int16))
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    peak_db = 20.0 * np.log10(max(peak, 1e-12))
    print(f"wrote transcribed.mid and aura_g_talkbox_demo.wav "
          f"({len(output) / SR:.1f} s, peak {peak_db:.1f} dBFS)")


if __name__ == "__main__":
    main()
