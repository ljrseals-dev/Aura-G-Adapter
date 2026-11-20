#pragma once
#include <array>
#include <atomic>
#include <cmath>
#include <vector>

/**
 * Standalone C++17 AURA-G waveguide/amp engine.
 * prepare() may allocate and must run off the audio thread.
 * processBlock() performs no allocation and owns all mutable DSP state per instance.
 */
class AuraGEngine {
public:
    static constexpr int kLatentDim = 32;
    static constexpr int kMaxDelay = 2048;
    static constexpr float kTwoPi = 6.28318530718f;

    // May throw std::invalid_argument / std::bad_alloc; call outside audio callback.
    void prepare(double sampleRate, int maxBlockSize);
    void reset() noexcept;

    void setAmpGain(float v) noexcept { ampGain.store(clampFinite(v, 0.0f, 1.0f, 0.5f)); }
    void setFxMix(float v) noexcept { fxMix.store(clampFinite(v, 0.0f, 1.0f, 0.3f)); }
    void setPhysicalMix(float v) noexcept { physMix.store(clampFinite(v, 0.0f, 1.0f, 0.2f)); }
    void setDrive(float v) noexcept { drive.store(clampFinite(v, 0.0f, 4.0f, 1.0f)); }
    void setSag(float v) noexcept { sag.store(clampFinite(v, 0.0f, 1.0f, 0.3f)); }
    void setSaturation(float v) noexcept { saturation.store(clampFinite(v, 0.0f, 1.0f, 0.7f)); }
    void setToneBass(float v) noexcept { toneBass.store(clampFinite(v, -12.0f, 12.0f, 0.0f)); }
    void setToneMid(float v) noexcept { toneMid.store(clampFinite(v, -12.0f, 12.0f, 0.0f)); }
    void setToneTreble(float v) noexcept { toneTreble.store(clampFinite(v, -12.0f, 12.0f, 0.0f)); }
    void setWaveguideFreq(float v) noexcept { wgFreq.store(clampFinite(v, 60.0f, 1200.0f, 196.0f)); }
    void setWaveguideDamp(float v) noexcept { wgDamp.store(clampFinite(v, 0.5f, 0.999f, 0.995f)); }
    void setExpression(int idx, float v) noexcept {
        if (idx >= 0 && idx < 8) expression[static_cast<size_t>(idx)].store(clampFinite(v, 0.0f, 1.0f, 0.0f));
    }
    void readLatent(float* dst) const noexcept;
    void writeLatent(const float* src) noexcept;
    void processBlock(float* const* channels, int numChannels, int numSamples) noexcept;

    struct Biquad {
        float b0=1.0f,b1=0.0f,b2=0.0f,a1=0.0f,a2=0.0f;
        float x1=0.0f,x2=0.0f,y1=0.0f,y2=0.0f;
        float process(float x) noexcept {
            const float w=x-a1*y1-a2*y2;
            const float y=b0*w+b1*x1+b2*x2;
            x2=x1; x1=w; y2=y1; y1=y; return y;
        }
        void reset() noexcept { x1=x2=y1=y2=0.0f; }
    };

private:
    static float clampFinite(float v,float lo,float hi,float fallback) noexcept {
        return std::isfinite(v) ? (v<lo?lo:(v>hi?hi:v)) : fallback;
    }
    void updateBiquads() noexcept;

    double sampleRate=48000.0;
    int maxBlock=512;
    bool prepared=false;
    std::atomic<float> ampGain{0.5f},fxMix{0.3f},physMix{0.2f},drive{1.0f},sag{0.3f},saturation{0.7f};
    std::atomic<float> toneBass{0.0f},toneMid{0.0f},toneTreble{0.0f},wgFreq{196.0f},wgDamp{0.995f};
    std::array<std::atomic<float>,8> expression{};
    std::array<std::atomic<float>,kLatentDim> latent{};

    Biquad bassEq,midEq,trebleEq;
    float lastBassDb=0.0f,lastMidDb=0.0f,lastTrebleDb=0.0f;
    float envState=0.0f;
    std::vector<float> delayLine;
    int delayLen=0,writePos=0;
    float lastSample=0.0f,apEta=0.0f,apIn=0.0f,apOut=0.0f;
    float fxState=0.0f;
    float hpX1=0.0f,hpY1=0.0f,lpY1=0.0f;
    static constexpr float kLoopFilterDelay=1.0f;
};
