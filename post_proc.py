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

def get_3_21(h, f1, f2):
   A3f1=h[np.argmin(np.abs(h[:,0]/(3*f1)-1))]
   A3f2=h[np.argmin(np.abs(h[:,0]/(3*f2)-1))]
   A2f1_f2=h[np.argmin(np.abs(h[:,0]/(2*f1+f2)-1))]
   Af1_2f2=h[np.argmin(np.abs(h[:,0]/(f1+2*f2)-1))]
   return np.array([A3f1, A3f2, A2f1_f2, Af1_2f2]).reshape((4,2))

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
   Lcos=np.cos(2*np.pi*f*ts-np.pi/2.)
   Lsin=np.sin(2*np.pi*f*ts-np.pi/2.)

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

def figs_2D(svg, var="F"):
    DCs=np.load("data/diag/DCs.npy")
    fs=np.load("data/diag/fs.npy")    
    
    H=4
    Ans = np.zeros([len(DCs), len(fs), H])
    Nhs = np.zeros([len(DCs), len(fs)])    

    for i, DC in enumerate(DCs):
       for j, f in enumerate(fs):
          print("DC = %.5f, f = %s"%(DC,f))
          ts  = np.load("data/diag/t_DC_%.05f_f_%.05f.npy"%(DC,f))
          X   = np.load("data/diag/X_DC_%.05f_f_%.05f.npy"%(DC,f))
          I1  = np.load("data/diag/I1_DC_%.05f_f_%.05f.npy"%(DC,f))
          if var=="F": Nhs[i,j], Ans[i,j] = get_meas(ts, X*I1, f, DC, th=.0001)    
          if var=="X": Nhs[i,j], Ans[i,j] = get_meas(ts, X, f, DC, th=.1)    
           
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
    pl.clf()
    #return fs, DCs, A1s

def fig_tr_sp(DC, f, svg, folder = "data/diag/", var="F"):
   tf=100/f
   N=100000
   sample_rate = N/tf      

   N_tr = int(4*N/100)   

   ts  = np.load(folder+"t_DC_%.05f_f_%.05f.npy"%(DC,f))
   X   = np.load(folder+"X_DC_%.05f_f_%.05f.npy"%(DC,f))   
   if var=="F": X=X*np.load(folder+"I1_DC_%.05f_f_%.05f.npy"%(DC,f))

   freqs, F_F = get_fft(X[-N//2:], sample_rate)
   hs = get_harmonics(freqs, F_F)
   hs = filter_nh(hs, f)      
   pl.figure(figsize=(10,6))
   pl.subplot(211)
   figs.plot_fluo(ts[-N_tr:], X[-N_tr:], title = "DC = %.5f, f = %.5f"%(DC, f))
   pl.subplot(212)
   figs.plot_FA(freqs, np.abs(F_F), hs, flim=20*f)
   pl.savefig(svg, bbox_inches="tight")
   pl.clf()

def fig_tr_sp_2f(DC, f1, f2, svg, folder = "data/diag/", var="F"):
   tf=100/min(f1,f2)
   N=100000
   sample_rate = N/tf      

   N_tr = int(4*N/100)   

   ts  = np.load(folder+"t_DC_%.05f_f1_%.05f_f2_%.05f.npy"%(DC,f1, f2))
   X   = np.load(folder+"X_DC_%.05f_f1_%.05f_f2_%.05f.npy"%(DC,f1, f2))   
   if var=="F": X=X*np.load(folder+"I1_DC_%.05f_f1_%.05f_f2_%.05f.npy"%(DC,f1, f2))

   freqs, F_F = get_fft(X[-N//2:], sample_rate)
   hs = get_harmonics(freqs, F_F)
   h1s = filter_nh(hs, f1)      
   h2s = filter_nh(hs, f2)      
   A321 = get_3_21(hs, f1, f2)
   
   pl.figure(figsize=(10,6))
   pl.subplot(211)
   figs.plot_fluo(ts[-N_tr:], X[-N_tr:], title = "DC = %.5f, f1 = %.5f, f2 = %.5f"%(DC, f1, f2))
   pl.subplot(212)
   figs.plot_FA(freqs, np.abs(F_F), h1s, h2s, A321, flim=5*max(f1,f2))
   pl.savefig(svg, bbox_inches="tight")
   pl.clf()

def get_A321(DC, f1, f2, folder = "data/diag/", var="F"):
   tf=100/min(f1,f2)
   N=100000
   sample_rate = N/tf      

   N_tr = int(4*N/100)   
   print(folder+"X_DC_%.05f_f1_%.05f_f2_%.05f.npy"%(DC,f1, f2))
   #ts  = np.load(folder+"t_DC_%.05f_f1_%.05f_f2_%.05f.npy"%(DC,f1, f2))
   X   = np.load(folder+"X_DC_%.05f_f1_%.05f_f2_%.05f.npy"%(DC,f1, f2))   
   if var=="F": X=X*np.load(folder+"I1_DC_%.05f_f1_%.05f_f2_%.05f.npy"%(DC,f1, f2))

   freqs, F_F = get_fft(X[-N//2:], sample_rate)
   hs = get_harmonics(freqs, F_F)
   A321 = get_3_21(hs, f1, f2)
   return A321

def get_A321s(folder, var = "F"):
    fs=np.load(folder+"fs.npy")    

    DC = .0015
    
    A321s = np.zeros([len(fs),len(fs),4]) 
    for i, f1 in enumerate(fs):
       for j, f2 in enumerate(fs):
          A321 = get_A321(DC, f1, f2, folder, var)
          A321s[i,j] = A321[:,1]
    return A321s      

def plot_A321(folder):
   A321s = get_A321s(folder)
   for i in range(4):
      for j in range(4):
          pl.subplot(4, 4,4*i+1+j)
          pl.loglog(A321s[:,:,i], A321s[:,:,j], "k.")
          if j==0:
            if i==0: pl.ylabel(r"$A_{3f_1}$")
            if i==1: pl.ylabel(r"$A_{3f_2}$")
            if i==2: pl.ylabel(r"$A_{2f_1+f_2}$")
            if i==3: pl.ylabel(r"$A_{f_1+2f_2}$")
            pl.yticks([.0001,.01,1])
          else: pl.yticks([])
          if i==3:
            if j==0: pl.xlabel(r"$A_{3f_1}$")
            if j==1: pl.xlabel(r"$A_{3f_2}$")
            if j==2: pl.xlabel(r"$A_{2f_1+f_2}$")
            if j==3: pl.xlabel(r"$A_{f_1+2f_2}$")
            pl.xticks([.0001,.01,1])
          else: pl.xticks([])             
   pl.savefig("A321.png",bbox_inches="tight")

def Bode_diag(svg, var="X"):
    fs=np.load("data/1D/fs.npy")    
    #fs = np.logspace(-3,2,100)
    DC = .0015
    A1s = np.zeros(len(fs))
    As = np.zeros(len(fs))
    phis = np.zeros(len(fs))
    crs = np.zeros(len(fs))
    cis = np.zeros(len(fs))
    
    for i, f in enumerate(fs):
       ts  = np.load("data/1D/t_DC_%.05f_f_%.05f.npy"%(DC,f))
       X   = np.load("data/1D/X_DC_%.05f_f_%.05f.npy"%(DC,f))
       I1  = np.load("data/1D/I1_DC_%.05f_f_%.05f.npy"%(DC,f))
       if var=="X": Ns, A = get_meas(ts, X, f, DC, th=.1)
       if var=="F": Ns, A = get_meas(ts, X*I1, f, DC, th=.1)
       if var=="X": crs[i], cis[i], As[i], phis[i] = get_AP(f, ts, X)
       if var=="F": crs[i], cis[i], As[i], phis[i] = get_AP(f, ts, X*I1)
       A1s[i] = A[0]
    pl.subplot(211)
    pl.loglog(fs, np.abs(crs), "k")
    #if var=="X": pl.ylabel(r"$|A_1(X)|$")
    #if var=="F": pl.ylabel(r"$|A_1(F)|$")
    if var=="X": pl.ylabel(r"$|Re(FFT(X))|$")
    if var=="F": pl.ylabel(r"$|Re(FFT(F))|$")
    pl.xlabel("Frequency [Hz]")
    pl.title("DC=0.0015")
    pl.subplot(212)
    pl.loglog(fs, np.abs(cis), "k")
    if var=="X": pl.ylabel(r"$|Im(FFT(X))|$")
    if var=="F": pl.ylabel(r"$|Im(FFT(F))|$")
    #if var=="X": pl.ylabel(r"$\phi_1(X)$")
    #if var=="F": pl.ylabel(r"$\phi_1(F)$")
    pl.savefig(svg, bbox_inches="tight")   
    pl.clf()
    return fs, A1s     

Bode_diag("bode_X_reim.png", var="X")
Bode_diag("bode_F_reim.png", var="F")

#plot_A321("data/2f_diag/")
#figs_2D("Harmonics_X.png", "X")
#figs_2D("Harmonics_F.png", "F")
#fig_tr_sp_2f(0.0015, 1, 5,"2f_tr_F.png",  "data/")


"""
DCs=np.load("/home/kodda/Data/dronpa2/data/diag/DCs.npy")
fs=np.load("/home/kodda/Data/dronpa2/data/diag/fs.npy")    #

fig_tr_sp(DCs[10], fs[1], "tr_F.png")
fig_tr_sp(DCs[10], fs[1], "tr_X.png", var="X")

Bode_diag("Bode_diag_F.png", "F")
Bode_diag("Bode_diag_X.png", "X")
"""

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