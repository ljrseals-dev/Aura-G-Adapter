# AURA-G current engineering status

**Assessment date:** 2026-11-18  
**Target branch:** `main`  
**Repair branch:** `repair/aura-g-processing-corrections`  
**State:** Experimental; source-level repair work in progress.

## Repository baseline

The baseline contains a Python AURA-G package, a separate adapter, guitar synthesis and MIDI helpers, a vocal-tract renderer, a synthetic demo, and a small unit-test suite. The adapter uses NumPy/SciPy. The complete processing pipeline also imports PyTorch and torchaudio.

The repository did not have a root `README.md` at the inspected baseline. `README_AURA_G.md` described the adapter and reported synthetic test results, but those historical results were not independently rerun during this repair session.

## Findings from source inspection

| Area | Baseline issue | Repair in this branch |
|---|---|---|
| Front end | Short clips could create negative frame counts; invalid or non-finite audio was not rejected consistently | Short input yields no frames; validates mono finite input and constructor parameters; corrected frame-count calculation and bounded YIN search |
| DDSP amplifier | Inference mutated `Parameter.data`, causing drive changes to accumulate between calls | Latent drive modulation is now a local forward-pass value; filter and nonlinear controls are bounded during processing |
| Pipeline | Empty/short clips could reach invalid latent aggregation; output path did not preserve short input | Empty clips return empty output; clips shorter than the analysis window are passed through; validates finite mono input; adds explicit transcription flush |
| Transcription | Active notes were appended on every call and had no end/duration lifecycle | Emits completed notes once, tracks frame time, records duration, and exposes `flush()` |
| MIDI | VLQ encoder accepted invalid negative deltas; parser read beyond truncated input and lacked structural bounds checks | Validates note/time/tempo/TPQ fields; bounds-checks track, event, VLQ, meta, and SysEx parsing; rejects unsupported SMPTE division |
| Blind extraction | Empty/short input and mismatched pairs were not handled; overlap-add did not normalize window accumulation | Validates training pairs and finite mono input; overlap-add uses squared-window normalization |
| Mixer | Invalid latent values could create negative or non-finite weights | Validates latent shape and finiteness; clamps and normalizes non-negative weights |
| Physical string | Fret affected damping but not fundamental pitch; damping could become negative; per-sample array rolls were costly | Recalculates delay for fretted pitch, bounds damping, and uses a circular delay-line cursor |
| Adapter/rendering | Empty note rendering and malformed audio/parameters were not handled consistently | Adds validation and safe empty results for empty audio/note collections |

## Work committed to the repair branch

- Reworked `aura_g/front_end.py`, `aura_g/ddsp_amp.py`, `aura_g/transcription.py`, and `aura_g/pipeline.py`.
- Hardened `aura_g/blind_extractor.py` and `aura_g/mixer.py`.
- Corrected fret-aware tuning in `aura_g/physical_model.py`.
- Hardened `harmonia_guitar.py` MIDI parsing/serialization and `aura_g_adapter.py` input/render handling.
- Expanded `test_aura_g_adapter.py` with regression cases for empty/short clips, YIN pitch, transcription lifecycle, MIDI malformed input, blind-extractor short pairs, and fret tuning.
- Added this status document and a repository-level README.

## Verification status

**Not yet executed in this environment.** The GitHub connector can create and inspect repository changes, but it does not provide a local Python runtime for running this test suite against the branch. No passing test result is claimed here.

The next verification step is to run:

```sh
python -m pip install numpy scipy
python -m unittest -v test_aura_g_adapter
```

The full pipeline additionally needs compatible PyTorch and torchaudio versions. CI should run the adapter tests and separately add an import/smoke test for the full pipeline once those dependencies are provisioned.

## Known remaining risks / follow-up work

1. Run the expanded tests and fix any regressions revealed by actual execution.
2. Test MIDI format 1 tempo maps across multiple tracks; the current parser uses a simple shared tempo and is not a complete tempo-map implementation.
3. Add polyphonic transcription or explicitly maintain the monophonic scope.
4. Validate on real guitar recordings across instruments, tunings, noise floors, and playing styles.
5. Add audio golden tests and listening review for the waveguide, DDSP, GRU, and tract paths.
6. Confirm GRU model training/checkpoint loading; randomly initialized weights are not a trained effects model.
7. Validate sample-rate conversion, stereo downmix policy, clipping behavior, and duration preservation in the demo.
8. Benchmark full-pipeline memory use, latency, and thread-safety before any real-time claim.

## Safety of repository changes

All changes are on `repair/aura-g-processing-corrections`. The existing `main` branch was not modified, force-pushed, or merged. The repair branch is intended for review and CI validation before integration.
