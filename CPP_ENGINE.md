# C++17 waveguide integration

This directory-level engine is an additional native processing path; it does not replace the Python AURA-G front end, transcription, MIDI, DDSP, guitar model, talk-box renderer, or their tests.

## Lifecycle and real-time boundary

- Call `AuraGEngine::prepare(sampleRate, maxBlockSize)` from a non-audio thread. It validates the sample rate/block size and allocates the fixed-capacity waveguide buffer. It may throw.
- Call `processBlock` only after successful preparation. It rejects null channel arrays, null channel pointers, non-positive channel/sample counts, and blocks larger than the prepared maximum.
- The delay buffer is allocated once in `prepare`. Pitch changes select a new active delay length and clear the already allocated storage; they do not resize or allocate in `processBlock`.
- FX, filter, envelope, and waveguide state belong to each engine instance. No static or thread-local mutable DSP state is used.
- Parameter setters are atomic snapshots. The engine does not promise coherent multi-parameter transactions.
- The callback does not lock, log, or allocate. The C++ regression test overrides global allocation operators and asserts zero allocations while processing, including a waveguide-frequency change.

## Validation

`g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. AuraGEngine.cpp tests/test_aura_g_engine.cpp -o /tmp/test_aura_g_engine && /tmp/test_aura_g_engine`

`python probe.py` builds the C ABI wrapper and measures six waveguide tuning points plus a decay check. GitHub Actions runs both checks alongside the existing Python adapter and full-pipeline suites.

These checks establish only the tested standalone C++ path. They do not establish JUCE plugin integration, host behavior, trained neural quality, real-guitar transcription accuracy, or production real-time suitability on target hardware.
