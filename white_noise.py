import numpy as np
import matplotlib.pyplot as pl
from scipy import signal
from scipy.integrate import odeint
from scipy import fft
from scipy.signal import find_peaks
import time

def get_fft(y, sample_rate):
   N= len(y)
   yf = fft.fft(y-y.mean())
   xf = fft.fftfreq(N, 1 / sample_rate)
   Nf=len(xf)
   xf = xf[:Nf//2]
   yf = yf[:Nf//2]
   return xf, yf

def get_local_max_2(x):
   return find_peaks(x, threshold=.000001)[0]

def get_harmonics(freqs, F_X):
   FA_X = np.abs(F_X)
   Ms_idxs = get_local_max_2(FA_X)   

   hs = np.zeros([len(Ms_idxs), 2])
   for i in range(len(Ms_idxs)):
      hs[i,0] = freqs[Ms_idxs[i]]
      hs[i,1] = FA_X[Ms_idxs[i]]
   return hs

def filter_nh(h, f):
   d=(h[:,0]/f)
   idxs = np.where((d>.1)*(d%1<.03)+((1-d%1)<.03)*(d>.1))[0]
   return h[idxs]

def filter_nh_th(h, th):
   idxs = np.where(h[:,1]>th)[0]
   return len(idxs)

def get_An(h, f, n):
    r = np.abs(h[:,0]/f-n)
    idx = np.argmin(r)
    if r[idx]<.1: return h[idx,1]
    else: return 0

def get_meas(ts, F, f, DC, H=4, plotit=False, th=10**(-4)):
    tf=100/f
    N=len(ts)
    sample_rate = N/tf       

    #cr, ci, amp, ph = get_AP(f, ts, F)
    freqs, F_F = get_fft(F[-N//2:], sample_rate)
    hs = get_harmonics(freqs, F_F)
    hs = filter_nh(hs, f)
    #Nh = hs.shape[0]  
    Nh = filter_nh_th(hs, th)  

    As = [get_An(hs, f, n) for n in range(1,H+1)]

    if plotit:
       N_show = int(4*sample_rate/f)
       pl.subplot(211)
       figs.plot_fluo(ts[-N_show:], F[-N_show:], title = 'DC=%.4f, f=%.4f'%(DC, f))
       pl.subplot(212)
       figs.plot_FA(freqs, np.abs(F_F), hs, 14*f)
       pl.savefig("test.png", bbox_inches='tight')
    return As


def get_wn(duration=10., fs=10000,f_low = .001, f_high = 1000):
   n_samples = int(duration * fs)+1   

   freqs = np.fft.fftfreq(n_samples, d=1/fs)
   spectrum = np.zeros(n_samples, dtype=complex)   

   band = np.logical_and(freqs >= f_low, freqs <= f_high)
   spectrum[band] = np.exp(1j * 2 * np.pi * np.random.rand(np.sum(band)))   

   band_neg = np.logical_and(freqs <= -f_low, freqs >= -f_high)
   spectrum[band_neg] = np.exp(1j * 2 * np.pi * np.random.rand(np.sum(band_neg)))

   band_limited_noise = np.fft.ifft(spectrum).real
   band_limited_noise /= np.max(np.abs(band_limited_noise))
   return band_limited_noise

def get_pars(DC=0.0001, f=1):
   pars = {"s12_1": 198,
           "s21_2": 415,
           "k21"  : .02,
           "I2"   : .0005,
           "I1_DC": .0001, #from .0001 to .002
           "I1_M" : .0001,
           "I1_f" : 1., 
           "X_tot": 1.      
           }   
   pars["I1_DC"] = DC        
   pars["I1_M"] = pars["I1_DC"]        
   pars["I1_f"] = f 
   pars['k2'] = pars["s21_2"]*pars["I2"]+pars["k21"]
   return pars

def get_stat(pars):
    return pars["k2"]*pars["X_tot"]/(pars["k2"]+pars["s12_1"]*pars["I1_DC"])

def dynamics(X, t, pars):
   I1 = W[int(t*fs)]
   dX = -pars['s12_1']*I1*X+pars['k2']*(pars["X_tot"]-X)
   return dX

t0=time.time()
fs = 10000
duration = 6000
N=duration*fs+1
ampl=0.0015
pars=get_pars(DC=ampl, f=1)

W = get_wn(duration*1.1, fs, f_low = .001, f_high = 100)*ampl
W-=min(W)

ts=np.linspace(0, duration, N)
X1_0 = get_stat(pars)
X1s=odeint(dynamics, X1_0, ts, args=(pars,), hmax=0.001, atol=1e-7, rtol=1e-11, mxstep=5000)

freqs, FF = get_fft(X1s.T[0]*W[:len(X1s.T[0])], fs)
freqs, FX = get_fft(X1s.T[0], fs)

fs=np.load("data/1D/fs.npy")    
DC = .0015
AXs = np.zeros(len(fs))
AFs = np.zeros(len(fs))
    
for i, f in enumerate(fs):
   ts  = np.load("data/1D/t_DC_%.05f_f_%.05f.npy"%(DC,f))
   X   = np.load("data/1D/X_DC_%.05f_f_%.05f.npy"%(DC,f))
   I1  = np.load("data/1D/I1_DC_%.05f_f_%.05f.npy"%(DC,f))
   AX = get_meas(ts, X, f, DC, th=.1)
   AF = get_meas(ts, X*I1, f, DC, th=.1)
   AXs[i] = AX[0]
   AFs[i] = AF[0]

pl.subplot(211)
pl.loglog(freqs, np.abs(FX), "r,")
pl.loglog(fs, AXs, "k")
#pl.loglog(freqs, np.abs(FX)/np.max(np.abs(FX)), "r,")
#pl.loglog(fs, AXs/np.max(AXs), "k")
pl.ylabel(r"$|A_1(X)|$")
pl.xlabel("Freq. [Hz]")
pl.subplot(212)
#pl.loglog(freqs, np.abs(FF)/np.max(np.abs(FF)), "r,")
#pl.loglog(fs, AFs/np.max(AFs), "k")
pl.loglog(freqs, np.abs(FF), "r,")
pl.loglog(fs, AFs, "k")
pl.ylabel(r"$|A_1(X)|$")
pl.xlabel("Freq. [Hz]")



#freqs, F = get_fft(W, fs)

print(time.time()-t0)