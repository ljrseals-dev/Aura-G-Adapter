# AURA-G Adapter

AURA-G is an experimental guitar-audio processing and transcription project. It combines a latent-control front end, monophonic pitch/onset analysis, guitar-string physical models, an optional DDSP amplifier and recurrent effects, MIDI utilities, and a talk-box-style vocal-tract renderer.

## Current project state

**Experimental / repair in progress.** The adapter's NumPy/SciPy path and the broader Torch-based pipeline have different dependency and validation requirements. This repository should not yet be treated as a production-ready real-time processor or as a validated polyphonic transcription system.

See [STATUS.md](STATUS.md) for the current engineering assessment, completed corrections, remaining risks, and verification limits. The adapter-specific design notes are in [README_AURA_G.md](README_AURA_G.md).

## Capabilities

- Monophonic audio-to-note transcription using the AURA-G front end and a corrected YIN implementation.
- Fractional-delay tuned waveguide guitar-string rendering.
- A 48-section Kelly–Lochbaum-style tract renderer.
- Standard MIDI File Type-0 writer and Type-0/1 reader with bounded event parsing.
- Experimental DDSP amplifier, GRU effects, latent bus, and physical guitar model.

## Installation

The adapter, MIDI helpers, and adapter tests require:

```sh
python -m pip install numpy scipy
```

The full `aura_g.pipeline.AURAG` path additionally requires compatible `torch` and `torchaudio` installations:

```sh
python -m pip install torch torchaudio
```

Install PyTorch and torchaudio versions matched to the target platform and accelerator from the official PyTorch installation selector.

## Run

Run the regression suite:

```sh
python -m unittest -v test_aura_g_adapter
```

Run the synthetic demonstration:

```sh
python demo_aura_g.py
```

Or provide a mono or multichannel WAV input (the demo downmixes multichannel input):

```sh
python demo_aura_g.py path/to/guitar.wav
```

The adapter API expects a finite, one-dimensional mono NumPy array. Use `aura_g_adapter.register()` to register the AURA-G transcriber with `midi_bridge.transcribe()`.

## Important limitations

- Pitch/onset transcription is monophonic; chords and overlapping same-pitch notes are not reliably separated.
- Synthetic pluck tests do not establish accuracy on real guitar recordings.
- The GRU effects are randomly initialized unless trained weights are supplied; this repository does not establish a trained model checkpoint.
- The physical guitar path follows latent pitch/onset blocks and retains waveguide/body-filter state across processing blocks. The full pipeline remains offline-oriented and has not been qualified for real-time latency, allocation behavior, or thread safety.
- The vocal-tract model is an experimental effect, not a validated speech synthesizer.
- Audio normalization, sample-rate conversion, and channel handling should be performed explicitly at application boundaries.

## Development status

See [STATUS.md](STATUS.md). All repairs in the current repair branch are proposed changes and should be merged only after CI and a human review of the generated audio and MIDI outputs.
