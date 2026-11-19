# AURA-G current engineering status

**Assessment date:** 2026-11-18  
**Target branch:** `main`  
**Repair branch:** `repair/aura-g-processing-corrections`  
**State:** Experimental; adapter regression suite passes on the repair branch; full-pipeline validation remains open.

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
| MIDI | VLQ encoder accepted invalid negative deltas; parser read beyond truncated input, lacked structural bounds checks, and applied tempo changes without a global format-1 timeline | Validates note/time/tempo/TPQ fields; bounds-checks track, event, VLQ, meta, and SysEx parsing; rejects unsupported SMPTE division; merges format-1 tempo events across tracks and keeps same-pitch notes on separate tracks independent |
| Blind extraction | Empty/short input and mismatched pairs were not handled; overlap-add did not normalize window accumulation | Validates training pairs and finite mono input; overlap-add uses squared-window normalization |
| Mixer | Invalid latent values could create negative or non-finite weights | Validates latent shape and finiteness; clamps and normalizes non-negative weights |
| Physical string | Fret affected damping but not fundamental pitch; damping could become negative; per-sample array rolls were costly; control types were weakly validated | Recalculates delay for fretted pitch, bounds damping, uses a circular delay-line cursor, and rejects invalid fret/sample-count/control values |
| Adapter/rendering | Empty note rendering and malformed audio/parameters were not handled consistently | Adds validation and safe empty results for empty audio/note collections |

## Work committed to the repair branch

- Reworked `aura_g/front_end.py`, `aura_g/ddsp_amp.py`, `aura_g/transcription.py`, and `aura_g/pipeline.py`.
- Hardened `aura_g/blind_extractor.py` and `aura_g/mixer.py`.
- Corrected fret-aware tuning and validated physical-model controls in `aura_g/physical_model.py`; removed redundant waveguide buffer initialization in the pipeline that was immediately overwritten by the pluck operation.
- Hardened `harmonia_guitar.py` MIDI parsing/serialization and `aura_g_adapter.py` input/render handling.
- Expanded `test_aura_g_adapter.py` with regression cases for empty/short clips, YIN pitch, transcription lifecycle, malformed MIDI, global format-1 tempo maps, overlapping same-pitch notes across tracks, blind-extractor short pairs, and fret tuning.
- Added `test_aura_g_pipeline.py` and a separate CPU PyTorch/torchaudio CI job to exercise the complete `AURAG.process()` path rather than inferring full-pipeline health from adapter tests alone.
- Added this status document and a repository-level README.

## Verification status

**Passed in GitHub Actions on 2026-11-19.** Run 44461295579 executed `python -m unittest -v test_aura_g_adapter` on commit `902d39139c7953372015530663183be637c1132c`: **12 tests passed**. The run reported 10/10 synthetic note matches (F1 1.00), 0.01-cent worst-case open/fretted tuning error, and 32.3 dB decay over one second for the T60=2 s test.

The first CI run exposed a real transcription defect: log-pitch value zero was treated as silence, although zero is the valid A4 (440 Hz) encoding. The transcription engine, adapter, and physical pipeline now decode A4 correctly; a dedicated regression test covers A4 followed by a dynamics-gated silent frame.

The adapter regression suite now passes with 14 tests, including a format-1 fixture that verifies a tempo change in the tempo track affects note timing in another track and that same-pitch notes in separate tracks do not overwrite each other. The separate CPU PyTorch/torchaudio pipeline job also passes its 3 smoke tests. This confirms the pipeline imports and executes for the covered synthetic cases; it is not evidence of trained-model quality or real-time performance.

## Known remaining risks / follow-up work

1. Extend MIDI coverage to malformed running-status cases and real-world format-1 files; the new global tempo-map behavior is covered by a focused fixture, not a broad MIDI corpus.
2. Evaluate the full pipeline against real guitar recordings and trained checkpoints; the smoke tests use a synthetic sine input and randomly initialized neural modules.
3. Add polyphonic transcription or explicitly maintain the monophonic scope.
4. Validate on real guitar recordings across instruments, tunings, noise floors, and playing styles.
5. Add audio golden tests and listening review for the waveguide, DDSP, GRU, and tract paths.
6. Confirm GRU model training/checkpoint loading; randomly initialized weights are not a trained effects model.
7. Validate sample-rate conversion, stereo downmix policy, clipping behavior, and duration preservation in the demo.
8. Benchmark full-pipeline memory use, latency, and thread-safety before any real-time claim.

## Safety of repository changes

All changes are on `repair/aura-g-processing-corrections`. The existing `main` branch was not modified, force-pushed, or merged. The repair branch is intended for review and CI validation before integration.

## Follow-up defect fixes (2026-11-19)

A deeper pass after the initial green smoke run found additional defects that were not covered by the original tests:

- MIDI note-on events for an already-active pitch on the same track/channel overwrote the first note. The parser now keeps a FIFO queue of overlapping voices and pairs each note-off with the oldest open voice.
- MIDI writer tempo values could overflow the Standard MIDI File 24-bit tempo field. The writer now validates the representable range and accepts NumPy integer TPQ values.
- The latent bus now validates its 32-value vector, sample rate, and hop; write methods reject non-finite/wrong-sized data; getters return copies rather than mutable views.
- The GRU effects path now bypasses conversion at equal sample rates and uses a bounded rational up/down ratio for torchaudio resampling. It validates batch/latent shapes, preserves the public sample count, and returns empty input safely.
- The fallback MIDI bridge now validates mono finite input, handles empty and exact-window clips, and bounds the YIN lag search. The adapter and fallback YIN estimators explicitly reject silent frames rather than reporting a spurious pitch.
- The physical string starts silent until plucked, rejects fundamentals above Nyquist, and validates body resonator modes and synthesis controls. The adapter rejects non-finite synthesis controls and malformed note timing/velocity before rendering.

### Follow-up verification

Run #44466498199 on the earlier follow-up commit had a passing full-pipeline job (3 tests) and an adapter job with 20 passing tests plus one failing newly authored MIDI overlap fixture. The failure was in the test fixture's end-of-track byte construction, not a parser assertion; the fixture was corrected to use explicit byte values. Additional regression cases were then added for silent YIN frames and unplucked-string silence.

**Latest verification passed on PR head 91d5ec859511a1292d5264db12241a99477416b0.** GitHub Actions run [#44466891104](https://github.com/ljrseals-dev/Aura-G-Adapter/actions/runs/44466891104) completed successfully: 25 adapter regression tests and 5 CPU PyTorch/torchaudio pipeline tests passed.
