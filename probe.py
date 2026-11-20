"""Compile and measure C++ waveguide tuning/decay: python probe.py (exit 0 = pass)."""
import ctypes
import os
import subprocess
import sys
import tempfile
import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__))
SO=os.path.join(tempfile.gettempdir(),"auragengine_probe.so")
subprocess.check_call(["g++","-O2","-std=c++17","-Wall","-Wextra","-Werror","-shared","-fPIC",
                       os.path.join(HERE,"AuraGEngine.cpp"),os.path.join(HERE,"wrap.cpp"),"-o",SO])
lib=ctypes.CDLL(SO)
lib.run.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_double]+[ctypes.c_float]*5
SR=48000
def render(x,wg=196.0,damp=0.995):
    x=np.ascontiguousarray(x,dtype=np.float32); y=np.zeros_like(x)
    lib.run(x.ctypes.data,y.ctypes.data,len(x),SR,wg,damp,0.0,0.0,1.0)
    return y.astype(float)

passed=True
for target in (82.41,196.0,330.0,440.0,659.26,880.0):
    x=np.zeros(SR,dtype=np.float32); x[:48]=0.5
    y=render(x,target)
    spectrum=np.abs(np.fft.rfft(y*np.hanning(len(y)),1<<19))
    frequencies=np.fft.rfftfreq(1<<19,1/SR)
    band=(frequencies>target*0.8)&(frequencies<target*1.2)
    peak=int(np.argmax(np.where(band,spectrum,0.0)))
    a,b,c=np.log(spectrum[peak-1:peak+2]+1e-30)
    hz=frequencies[peak]+0.5*(a-c)/(a-2*b+c)*(frequencies[1]-frequencies[0])
    cents=1200*np.log2(hz/target)
    print(f"target {target:7.2f} Hz measured {hz:7.2f} Hz ({cents:+.2f} cents)")
    passed &= abs(cents)<1.0 and bool(np.all(np.isfinite(y)))

time=np.arange(SR*3)/SR
x=np.zeros(SR*3,dtype=np.float32); x[:2400]=(0.5*np.sin(2*np.pi*196*time[:2400])).astype(np.float32)
y=render(x,damp=0.98)
rms=lambda lo,hi: np.sqrt(np.mean(y[int(lo*SR):int(hi*SR)]**2))+1e-18
decay_db=20*np.log10(rms(0.2,0.25)/rms(1.0,1.05))
print(f"decay 0.2s to 1.0s at damping 0.98: {decay_db:.1f} dB")
passed &= decay_db>15.0 and bool(np.all(np.isfinite(y)))
print("PASS" if passed else "FAIL")
sys.exit(0 if passed else 1)
