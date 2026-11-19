# AURA-G adapter

The adapter connects AURA-G's NumPy/SciPy front end to the guitar / talk-box pipeline. The adapter path requires NumPy and SciPy; the complete `aura_g.pipeline` additionally requires PyTorch and torchaudio.

- `aura_g_transcribe(audio, sr)`: AURA-G's front end -> monophonic notes with estimated durations. The adapter uses a first-CMND-threshold-crossing YIN implementation to reduce octave errors and rebuilds note spans from the pitch/onset tracks.
- `TunedWaveguideString`: fractional-delay waveguide synthesis with fret-aware fundamental tuning and a configurable T60 decay.
- `render_talkbox_aura(notes)`: plucked-string source -> 48-section tract model. `register()` registers the AURA-G transcriber with `midi_bridge.transcribe()`.
- `harmonia_guitar.py`: Karplus-Strong plucks and Standard MIDI File Type-0 helpers.

## Validation scope

The existing tests use synthetic monophonic guitar plucks. Historical results recorded before the repair branch (including the reported 10-note match and tuning/decay measurements) are project notes, not independently reproduced results from this repair session.

The expanded regression suite now includes empty/short input, YIN pitch, transcription event lifecycle, malformed MIDI data, short blind-extraction training pairs, and fret-aware physical-model checks. Run it with:

```sh
python -m unittest -v test_aura_g_adapter
```

## Known limitations

- Monophonic YIN is not polyphonic transcription; chords and overlapping notes are not separated reliably.
- Synthetic test signals do not establish performance on real guitar recordings.
- DDSP and GRU processing are experimental; no trained GRU checkpoint is supplied or established by these tests.
- The full pipeline has not been qualified for real-time latency, thread safety, or allocation behavior.
- The tract renderer is an experimental effect, not a validated speech synthesizer.

See the repository [README](README.md) and [current engineering status](STATUS.md) for setup and repair status.
