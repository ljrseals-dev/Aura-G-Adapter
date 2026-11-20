#include "AuraGEngine.h"
#include <algorithm>
#include <stdexcept>

void AuraGEngine::prepare(double sr,int maxBlockSize) {
    if (!std::isfinite(sr) || sr < 8000.0 || sr > 192000.0)
        throw std::invalid_argument("sample rate must be finite and in [8000, 192000]");
    if (maxBlockSize <= 0)
        throw std::invalid_argument("maxBlockSize must be positive");
    sampleRate=sr;
    maxBlock=maxBlockSize;
    // Allocate the complete delay capacity once, away from the audio callback.
    delayLine.assign(static_cast<size_t>(kMaxDelay),0.0f);
    delayLen=0;
    prepared=true;
    reset();
    lastBassDb=lastMidDb=lastTrebleDb=0.0f;
    updateBiquads();
}

void AuraGEngine::reset() noexcept {
    std::fill(delayLine.begin(),delayLine.end(),0.0f);
    writePos=0; lastSample=0.0f; apIn=apOut=apEta=0.0f;
    bassEq.reset(); midEq.reset(); trebleEq.reset();
    envState=0.0f; fxState=0.0f;
    hpX1=hpY1=lpY1=0.0f;
}

static void designPeak(AuraGEngine::Biquad& q,float sr,float f0,float Q,float gainDb) noexcept {
    const float A=std::pow(10.0f,gainDb/40.0f);
    const float w0=AuraGEngine::kTwoPi*f0/sr;
    const float alpha=std::sin(w0)/(2.0f*Q),cw=std::cos(w0);
    const float b0=1.0f+alpha*A,b1=-2.0f*cw,b2=1.0f-alpha*A;
    const float a0=1.0f+alpha/A,a1=-2.0f*cw,a2=1.0f-alpha/A;
    q.b0=b0/a0;q.b1=b1/a0;q.b2=b2/a0;q.a1=a1/a0;q.a2=a2/a0;
}

void AuraGEngine::updateBiquads() noexcept {
    const float b=toneBass.load(std::memory_order_relaxed);
    const float m=toneMid.load(std::memory_order_relaxed);
    const float t=toneTreble.load(std::memory_order_relaxed);
    if(b!=lastBassDb){designPeak(bassEq,static_cast<float>(sampleRate),100.0f,0.7f,b);lastBassDb=b;}
    if(m!=lastMidDb){designPeak(midEq,static_cast<float>(sampleRate),800.0f,1.0f,m);lastMidDb=m;}
    if(t!=lastTrebleDb){designPeak(trebleEq,static_cast<float>(sampleRate),3000.0f,0.7f,t);lastTrebleDb=t;}
}

void AuraGEngine::readLatent(float* dst) const noexcept {
    if(!dst)return;
    for(int i=0;i<kLatentDim;++i)dst[i]=latent[static_cast<size_t>(i)].load(std::memory_order_relaxed);
}
void AuraGEngine::writeLatent(const float* src) noexcept {
    if(!src)return;
    for(int i=0;i<kLatentDim;++i){
        const float v=src[i];
        latent[static_cast<size_t>(i)].store(std::isfinite(v)?v:0.0f,std::memory_order_relaxed);
    }
}

void AuraGEngine::processBlock(float* const* channels,int numChannels,int numSamples) noexcept {
    if(!prepared || delayLine.size()!=static_cast<size_t>(kMaxDelay) ||
       !channels || numChannels<=0 || numSamples<=0 || numSamples>maxBlock)return;
    for(int c=0;c<numChannels;++c)if(!channels[c])return;

    updateBiquads();
    const float gAmp=ampGain.load(std::memory_order_relaxed);
    const float gFx=fxMix.load(std::memory_order_relaxed);
    const float gPhys=physMix.load(std::memory_order_relaxed);
    const float drv=drive.load(std::memory_order_relaxed);
    const float sg=sag.load(std::memory_order_relaxed);
    const float sat=saturation.load(std::memory_order_relaxed);
    const float requested=wgFreq.load(std::memory_order_relaxed);
    const float fWg=std::min(requested,static_cast<float>(sampleRate*0.25));
    const float dWg=wgDamp.load(std::memory_order_relaxed);
    const float sum=gAmp+gFx+gPhys+1.0e-6f;
    const float wAmp=gAmp/sum,wFx=gFx/sum,wPhy=gPhys/sum;

    const float totalDelay=static_cast<float>(sampleRate)/fWg-kLoopFilterDelay;
    const int wanted=std::max(2,std::min(kMaxDelay-2,static_cast<int>(std::floor(totalDelay))));
    const float frac=totalDelay-static_cast<float>(wanted);
    apEta=(1.0f-frac)/(1.0f+frac);
    if(wanted!=delayLen){
        // Bounded clear only: capacity was allocated in prepare(); no resize/allocation here.
        std::fill(delayLine.begin(),delayLine.end(),0.0f);
        delayLen=wanted;writePos=0;lastSample=0.0f;apIn=apOut=0.0f;
    }

    for(int i=0;i<numSamples;++i){
        float x=0.0f;
        for(int c=0;c<numChannels;++c)x+=channels[c][i];
        x/=static_cast<float>(numChannels);
        float pre=std::tanh(drv*x);
        pre=bassEq.process(pre);pre=midEq.process(pre);pre=trebleEq.process(pre);
        const float absPre=std::fabs(pre);
        envState=0.999f*envState+0.001f*absPre;
        const float gainReduction=1.0f/(1.0f+sg*envState);
        const float satPre=pre*gainReduction*sat;
        float ampOut=satPre-(satPre*satPre*satPre)*0.3333333333f;
        ampOut=std::max(-1.0f,std::min(1.0f,ampOut));
        const float hpOut=0.995f*(hpY1+ampOut-hpX1);
        hpX1=ampOut;hpY1=hpOut;
        const float lpOut=lpY1+0.5f*(hpOut-lpY1);
        lpY1=lpOut;

        const float delayed=delayLine[static_cast<size_t>(writePos)];
        const float filtered=0.5f*(delayed+lastSample);
        const float ap=apEta*filtered+apIn-apEta*apOut;
        apIn=filtered;apOut=ap;
        delayLine[static_cast<size_t>(writePos)]=ap*dWg+lpOut*0.05f;
        writePos=(writePos+1)%delayLen;
        lastSample=filtered;
        const float physOut=delayed;

        // FX memory is per engine instance, never static/thread-local.
        fxState=0.7f*fxState+0.3f*lpOut;
        const float y=std::max(-1.0f,std::min(1.0f,wAmp*lpOut+wFx*fxState+wPhy*physOut));
        for(int c=0;c<numChannels;++c)channels[c][i]=y;
    }
}
