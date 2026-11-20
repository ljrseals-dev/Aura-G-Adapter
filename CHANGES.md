# AuraGEngine fixes (plugin/AuraGEngine.h, AuraGEngine.cpp)

1. **Did not compile**: the .cpp called `juce::jlimit` without including JUCE, and `designPeak` used the private `Biquad`. Added a local `juce::jlimit` fallback and made `Biquad` public.
2. **Waveguide tuning**: delay was `round(sr / f)` with no allowance for the loop filter's 1-sample delay: 7 to 15 cents flat (196 Hz: -7.2, 440 Hz: -14.5). Now a ring buffer plus a first-order all-pass fractional delay: within 0.4 cents from 82 to 880 Hz.
3. **Waveguide never decayed**: `wgDamp` only scaled the output, so the loop had unity gain (about 2.7 dB of decay in 2.3 s). It is now the per-loop gain: 0.98 gives -55 dB after 1 s, 0.995 about -29 dB.

Not changed: the DDSP amp, FX path and mixer (placeholder FX), and the JUCE plugin files, which were not built (no JUCE here).
Run `python probe.py`.
