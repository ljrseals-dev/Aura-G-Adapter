#include "AuraGEngine.h"
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <new>
#include <stdexcept>

static std::atomic<bool> trackAllocations{false};
static std::atomic<unsigned> allocationCount{0};
void* operator new(std::size_t n) {
    if(trackAllocations.load(std::memory_order_relaxed)) allocationCount.fetch_add(1,std::memory_order_relaxed);
    if(void* p=std::malloc(n)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) {
    if(trackAllocations.load(std::memory_order_relaxed)) allocationCount.fetch_add(1,std::memory_order_relaxed);
    if(void* p=std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
void operator delete[](void* p,std::size_t) noexcept { std::free(p); }

static void test_unprepared_and_invalid_inputs_are_safe() {
    AuraGEngine e;
    float x[16]={}; float* ch[]={x};
    e.processBlock(ch,1,16);
    bool threw=false; try { e.prepare(0.0,64); } catch(const std::invalid_argument&) { threw=true; }
    assert(threw);
    threw=false; try { e.prepare(48000.0,0); } catch(const std::invalid_argument&) { threw=true; }
    assert(threw);
}
static void test_processing_is_allocation_free_and_instance_isolated() {
    AuraGEngine a,b; a.prepare(48000,128); b.prepare(48000,128);
    a.setAmpGain(0.0f);a.setFxMix(1.0f);a.setPhysicalMix(0.0f);
    b.setAmpGain(0.0f);b.setFxMix(1.0f);b.setPhysicalMix(0.0f);
    float xa[128]={},xb[128]={}; xa[0]=0.8f;xb[0]=0.0f;
    float* ca[]={xa};float* cb[]={xb};
    allocationCount.store(0,std::memory_order_relaxed);
    trackAllocations.store(true,std::memory_order_relaxed);
    a.processBlock(ca,1,128);
    // Change delay length while processing: the preallocated capacity must suffice.
    a.setWaveguideFreq(440.0f);
    a.processBlock(ca,1,128);
    b.processBlock(cb,1,128);
    trackAllocations.store(false,std::memory_order_relaxed);
    assert(allocationCount.load(std::memory_order_relaxed)==0);
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
int main(){test_unprepared_and_invalid_inputs_are_safe();test_processing_is_allocation_free_and_instance_isolated();test_multichannel_and_null_guards();}
