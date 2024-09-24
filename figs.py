import numpy as np
import matplotlib.pyplot as pl

def plot_fluo(ts, Xs, title = None, svg = None):
    pl.plot(ts, Xs, "k")
    pl.xlabel("time [s]")
    pl.ylabel("F")
    if title: pl.title(title)
    if svg: pl.savefig(svg, bbox_inches="tight")

def plot_FA(freqs, FA, hs=[], flim=None, title = None, svg = None):
    pl.semilogy(freqs, FA, "k")
    pl.xlabel("frequency [Hz]")
    pl.ylabel(r"log $|A_F(F)|$")
    if flim: pl.xlim([0,flim])
    pl.ylim([0.0000001,5*max(FA)])
    if len(hs): pl.plot(hs[:,0], hs[:,1], "ro")
    if svg: pl.savefig(svg, bbox_inches="tight")

def fig_2D(X, fs=[], DCs=[], title=None, svg=None, ax=None, cm=pl.cm.jet, yt=True):
   #f= pl.figure(figsize=(12,8)) 
   if not(len(DCs)): DCs = np.linspace(0.0001, 0.002, 20)
   if not(len(fs)): 
      fs_v  = np.array([0.001, 0.01, 0.1, 10, 100])
      #fs = [r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$', r'$10$',r'$100$']
   if not(ax): ax = pl.subplot(111)
   im = ax.imshow(X, origin="lower", interpolation="nearest", cmap=cm)
   ax.set_xticks(np.arange(len(fs))[::8], labels=fs[::8])
   if yt: ax.set_yticks(np.arange(len(DCs))[::5], labels=np.round(DCs[::5], 4))
   ax.set_xlabel("Frequency [Hz]")
   ax.set_ylabel("DC")
   ax.set_title(title)
   pl.colorbar(im, ax=ax)
   if svg: pl.savefig(svg, bbox_inches="tight")
   #pl.clf()

def get_bode(hs):
   f= pl.figure(figsize=(12,8)) 
   if not(DCs): DCs = np.linspace(0.0001, 0.002, 20)
   if not(fs): 
      fs_v  = np.array([0.001, 0.01, 0.1, 10, 100])
      fs = [r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$', r'$10$',r'$100$']
   A1 = np.zeros([len(DCs), len(fs)])
   for i in range(len(DCs)):
      for j in range(len(fs)):
         A1[i,j] = hs[i][j][0][1]
   ax = pl.subplot(111)
   im = ax.imshow(A1, origin="lower", interpolation="nearest", cmap=pl.cm.jet)
   ax.set_xticks(np.arange(len(fs)), labels=fs)
   ax.set_yticks(np.arange(len(DCs))[::3], labels=np.round(DCs[::3], 4))
   ax.set_xlabel("Frequency [Hz]")
   ax.set_ylabel("DC")
   ax.set_title("A1")
   pl.colorbar(im, ax=ax)
   pl.savefig("A1.png", bbox_inches="tight")
   pl.clf()