# AURA-G adapter

Hooks AURA-G (numpy parts) into the guitar / talk-box pipeline. Needs numpy + scipy only; `aura_g/` is your code, unmodified.

- `aura_g_transcribe(audio, sr)`: AURA-G's front end -> notes with durations. Fixes: (1) `yin_pitch` takes the lowest cmnd minimum, which gives
  octave-low pitches on plucks, so the adapter patches in a first-below-threshold YIN; (2) the stock TranscriptionEngine emits one event per
  onset frame (about 8 per note) and no durations, so notes are rebuilt from the front end's pitch/onset tracks.
  Test: 10 of 10 notes matched, F1 1.00, durations 0.40 s (true 0.40); the stock engine gave 79 events (F1 0.22).
- `TunedWaveguideString`: replaces `DigitalWaveguideString`, which was +2.8 to +33.5 cents sharp (integer delay), ignored `fret` for pitch,
  and damped the loop by 1-0.05*fret (fret 5: -50 dB after 0.1 s; fret 20: silent). Tuned string: 0.01 cents, fret-5 pitch correct, T60 respected.
- `render_talkbox_aura(notes)`: string -> corrected 17 cm tract. `register()` makes `midi_bridge.transcribe()` use AURA-G.
- Not covered: DDSP amp, RNN FX, blind extractor, mixer (torch parts), polyphonic chords (YIN is monophonic), and real guitar recordings (tests use synthetic strings).

Run: `python -m unittest -v test_aura_g_adapter` and `python demo_aura_g.py`.
