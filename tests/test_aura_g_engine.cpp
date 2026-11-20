#include "AuraGEngine.h"
#include <cassert>
#include <cmath>
#include <stdexcept>

static void test_unprepared_and_invalid_inputs_are_safe() {
    AuraGEngine e;
    float x[16]={}; float* ch[]={x};
    e.processBlock(ch,1,16);
    bool threw=false; try { e.prepare(0.0,64); } catch(const std::invalid_argument&) { threw=true; }
    assert(threw);
    threw=false; try { e.prepare(48000.0,0); } catch(const std::invalid_argument&) { threw=true; }
    assert(threw);
}
static void test_processing_and_instance_isolation() {
    AuraGEngine a,b; a.prepare(48000,128); b.prepare(48000,128);
    a.setAmpGain(0.0f);a.setFxMix(1.0f);a.setPhysicalMix(0.0f);
    b.setAmpGain(0.0f);b.setFxMix(1.0f);b.setPhysicalMix(0.0f);
    float xa[128]={},xb[128]={}; xa[0]=0.8f;xb[0]=0.0f;
    float* ca[]={xa};float* cb[]={xb};
    a.processBlock(ca,1,128);
    b.processBlock(cb,1,128);
    // The silent instance must not inherit the first instance's FX state.
    for(float v:xb)assert(std::fabs(v)<1.0e-7f);
    for(float v:xa)assert(std::isfinite(v));
}
static void test_multichannel_and_null_guards() {
    AuraGEngine e;e.prepare(48000,64);
    float l[64]={},r[64]={};l[0]=0.5f;r[0]=-0.5f;
    float* channels[]={l,r};e.processBlock(channels,2,64);
    for(int i=0;i<64;++i)assert(std::isfinite(l[i])&&l[i]==r[i]);
    float* bad[]={nullptr};e.processBlock(bad,1,64);
    e.processBlock(nullptr,1,64);
    e.processBlock(channels,2,65);
}
int main(){test_unprepared_and_invalid_inputs_are_safe();test_processing_and_instance_isolation();test_multichannel_and_null_guards();}
