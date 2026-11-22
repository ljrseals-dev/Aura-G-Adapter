# AURA-G adapter + guitar chain

numpy/scipy only. `aura_g/` is your code, unmodified.

## Pieces
- `aura_g_adapter.py`: `aura_g_transcribe` (AURA-G front end -> notes with durations; patches its octave-low YIN and de-duplicates the per-frame onsets),
  `TunedWaveguideString` (replaces `DigitalWaveguideString`), `render_talkbox_aura`.
- `guitar_chain.py`: transcription -> tuned string -> talk-box tract -> LA-2A leveler -> loudness match -> true-peak limiter -> ProDither 16-bit (44.1 kHz).
- `guitar_gates.py`: 9 release gates (all pass). `test_aura_g_adapter.py`: 5 tests.
- Dependencies from the AURA-V work: `jowiphon_noise.py`, `prodither_fixed.py`, `la2a_py.py`, `the_order_aura.py` + `the_order_vocal_chain.py` (limiter and dither blocks), `vantage_os.py`.

## Results (synthetic plucked notes, not recordings of a real guitar)
- Transcription: 10/10 notes, F1 1.00, durations 0.40 s (stock engine: 79 events, F1 0.22).
- String: 0.01 cents tuning error (stock: +2.8 to +33.5 cents), fret changes pitch, decay follows T60.
- Chain: -14.06 LUFS, -1.05 dBTP, 100% on the 16-bit grid, 1-5 kHz noise floor 17.6 dB below flat TPDF, NaN/Inf-safe, bit-identical reruns.
- The talk-box output has 30 dB crest factor (each pluck's attack peaks about 5 ms in). The LA-2A (Peak Reduction 0.30) cuts that to 20 dB and narrows the
  note-to-note spread from 14.8 to 4.6 dB; the limiter then needs 6.5 dB of gain reduction to reach -14 LUFS (22 dB without the leveler).

## Not covered
Chords (the transcriber is monophonic), real guitar recordings, AURA-G's DDSP amp / RNN FX / blind extractor / mixer (torch), listening tests, WER.
`python -m unittest -v test_aura_g_adapter`, `python guitar_gates.py`, `python demo_aura_g.py`.
