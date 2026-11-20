#include "AuraGEngine.h"
extern "C" void run(const float* in,float* out,int n,double sr,float wg,float damp,float amp,float fx,float phys){
  AuraGEngine e; e.prepare(sr,512); e.setWaveguideFreq(wg); e.setWaveguideDamp(damp); e.setAmpGain(amp); e.setFxMix(fx); e.setPhysicalMix(phys);
  for(int i=0;i<n;i+=512){ int m=(n-i<512)?(n-i):512; float buf[512]; for(int k=0;k<m;++k) buf[k]=in[i+k]; float* ch[1]={buf}; e.processBlock(ch,1,m); for(int k=0;k<m;++k) out[i+k]=buf[k]; } }
