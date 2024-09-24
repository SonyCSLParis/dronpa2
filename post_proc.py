import numpy as np
from scipy import fft
import matplotlib.pyplot as pl
import figs
from scipy.signal import find_peaks

def get_fft(y, sample_rate):
   N= len(y)
   yf = fft.fft(y-y.mean())
   xf = fft.fftfreq(N, 1 / sample_rate)
   Nf=len(xf)
   xf = xf[:Nf//2]
   yf = yf[:Nf//2]
   return xf, yf

def get_local_max(x):
    return (np.diff(np.sign(np.diff(x))) < 0).nonzero()[0] + 1 

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

def get_pars(name):
    name = name.split('/')[-1].strip(".npy")
    res = name.split('_')
    DC = float(res[2])
    f = float(res[4])
    return DC, f

def get_ratio(F, I1):
    N=len(F)
    return (F[-N//3:].max()-F[-N//3:].min())/(I1[-N//3:].max()-I1[-N//3:].min())

def get_AP(f, ts, F):
   Lcos=np.cos(2*np.pi*f*ts)
   Lsin=np.sin(2*np.pi*f*ts)

   Fn=F-F.mean()

   cr=Fn@Lcos
   ci=Fn@Lsin
   cr*=2*(ts[1]-ts[0])/(ts[-1]-ts[0])
   ci*=2*(ts[1]-ts[0])/(ts[-1]-ts[0])
   
   amp=np.sqrt(cr**2+ci**2)
   ph=np.arctan(ci/cr)
   return cr, ci, amp, ph

def get_meas(ts, F, f, DC, H=4, plotit=False, th=10**(-4)):
    tf=100/f
    N=100000
    sample_rate = N/tf       

    cr, ci, amp, ph = get_AP(f, ts, F)
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
    return Nh, As

def figs_2D(svg):
    DCs=np.load("data/diag/DCs.npy")
    fs=np.load("data/diag/fs.npy")    
    
    H=4
    Ans = np.zeros([len(DCs), len(fs), H])
    Nhs = np.zeros([len(DCs), len(fs)])    

    for i, DC in enumerate(DCs):
       for j, f in enumerate(fs):
          print("DC = %.5f, f = %s"%(DC,f))
          ts  = np.load("data/diag/t_DC_%.5f_f_%s.npy"%(DC,f))
          X   = np.load("data/diag/X_DC_%.5f_f_%s.npy"%(DC,f))
          I1  = np.load("data/diag/I1_DC_%.5f_f_%s.npy"%(DC,f))
          Nhs[i,j], Ans[i,j] = get_meas(ts, X, f, DC, th=.1)    
    pl.figure(figsize=(25,4))
    ax = pl.subplot(151)
    figs.fig_2D(Nhs, np.round(fs,4), DCs, title="N harmonics", ax=ax, cm=pl.cm.jet)
    ax = pl.subplot(152)
    figs.fig_2D(Ans[:,:,0], np.round(fs,4), DCs, title="A1", ax=ax, cm=pl.cm.gray,yt=False)
    ax = pl.subplot(153)
    figs.fig_2D(Ans[:,:,1], np.round(fs,4), DCs, title="A2", ax=ax, cm=pl.cm.gray,yt=False)
    ax = pl.subplot(154)
    figs.fig_2D(Ans[:,:,2], np.round(fs,4), DCs, title="A3", ax=ax, cm=pl.cm.gray,yt=False)
    ax = pl.subplot(155)
    figs.fig_2D(Ans[:,:,3], np.round(fs,4), DCs, title="A4", ax=ax, cm=pl.cm.gray,yt=False)
    pl.savefig(svg, bbox_inches="tight")

    #return fs, DCs, A1s

def fig_tr_sp(DC, f, svg):
   tf=100/f
   N=100000
   sample_rate = N/tf      

   N_tr = int(4*N/100)   

   ts  = np.load("data/diag/t_DC_%.5f_f_%s.npy"%(DC,f))
   F   = np.load("data/diag/X_DC_%.5f_f_%s.npy"%(DC,f))   

   freqs, F_F = get_fft(F[-N//2:], sample_rate)
   hs = get_harmonics(freqs, F_F)
   hs = filter_nh(hs, f)      
   pl.subplot(211)
   figs.plot_fluo(ts[-N_tr:], F[-N_tr:], title = "DC = %.5f, f = %.5f"%(DC, f))
   pl.subplot(212)
   figs.plot_FA(freqs, np.abs(F_F), hs, flim=20*f)
   pl.savefig(svg, bbox_inches="tight")

def Bode_diag():
    fs=np.load("data/1D/fs.npy")    
    #fs = np.logspace(-3,2,100)
    DC = .001
    A1s = np.zeros(len(fs))
    As = np.zeros(len(fs))
    
    for i, f in enumerate(fs):
       print("data/1D/t_DC_%.5f_f_%s.npy"%(DC,f))
       ts  = np.load("data/1D/t_DC_%.5f_f_%s.npy"%(DC,f))
       F   = np.load("data/1D/F_DC_%.5f_f_%s.npy"%(DC,f))
       I1  = np.load("data/1D/I1_DC_%.5f_f_%s.npy"%(DC,f))
       cr, ci, amp, ph = get_AP(f, ts[-5000:], F[-5000:]/I1[-5000:])
       A1s[i] = amp
       As[i] = get_A(F/I1, I1)
    return fs, A1s, As     

figs_2D("Harmonics_X.png")

#fs, A1s, As = Bode_diag()
#fs, DCs, A1s = figs_2D()

#DCs=np.load("/home/kodda/Data/dronpa2/data/diag/DCs.npy")
#fs=np.load("/home/kodda/Data/dronpa2/data/diag/fs.npy")    #

#f  = fs[18]
#DC = DCs[5]#

#tf=100/f
#N=100000
#sample_rate = N/tf       #

#X   = np.load("data/diag/X_DC_%.5f_f_%s.npy"%(DC,f))
#I1  = np.load("data/diag/I1_DC_%.5f_f_%s.npy"%(DC,f))#

#F=X*I1          
#freqs, F_F = get_fft(F[-N//2:], sample_rate)
#hs = get_harmonics(freqs, F_F)
#hs = filter_nh(hs, f)
#r = hs[:,0]/f-2
#idx = np.argmin(r)#

#A1 = get_An(hs, f, 1)
#A2 = get_An(hs, f, 2)
#A3 = get_An(hs, f, 3)


#name = "/home/kodda/Data/dronpa2/data/diag/t_DC_0.00190_f_0.12742749857031335.npy"
#DC, f = get_pars(name)

"""
DCs=np.load("/home/kodda/Data/dronpa2/data/diag/DCs.npy")
fs=np.load("/home/kodda/Data/dronpa2/data/diag/fs.npy")    

f  = fs[0]
DC = DCs[18]
svg = "test_X.png"
fig_tr_sp(DC, f, svg)

figs_2D()
"""