#include "AuraGEngine.h"
extern "C" void run(const float* in,float* out,int n,double sr,float wg,float damp,float amp,float fx,float phys) {
    if(!in || !out || n<=0) return;
    AuraGEngine engine;
    engine.prepare(sr,512);
    engine.setWaveguideFreq(wg);
    engine.setWaveguideDamp(damp);
    engine.setAmpGain(amp);
    engine.setFxMix(fx);
    engine.setPhysicalMix(phys);
    for(int i=0;i<n;i+=512) {
        const int count=(n-i<512)?(n-i):512;
        float buffer[512];
        for(int k=0;k<count;++k) buffer[k]=in[i+k];
        float* channel[]={buffer};
        engine.processBlock(channel,1,count);
        for(int k=0;k<count;++k) out[i+k]=buffer[k];
    }
}
