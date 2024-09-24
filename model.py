import numpy as np
from scipy.integrate import odeint
from scipy import fft
import matplotlib.pyplot as pl
import pickle
import figs
import time
import json
from joblib import Parallel, delayed
import os

def get_I1(pars, t):
    return (pars["I1_DC"]+pars["I1_M"]*np.sin(2*np.pi*pars["I1_f"]*t))

def dynamics(X, t, pars):
   I1=get_I1(pars, t)
   dX1 = -pars['s12_1']*I1*X+(pars['k2']+pars["s21_2"]*pars["I2"])*(pars["X_tot"]-X)
   return dX1

def get_stat(pars):
    return pars["k2"]*pars["X_tot"]/(pars["k2"]+pars["s12_1"]*pars["I1_DC"])

def get_fft(y, sample_rate):
   N= len(y)
   yf = fft.fft(y-y.mean())
   xf = fft.fftfreq(N, 1 / sample_rate)
   Nf=len(xf)
   xf = xf[:Nf//2]
   yf = yf[:Nf//2]
   return xf, yf

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

def sim(DC, f, svg_folder="data/"):
   print(svg_folder+"I1_DC_%.05f_f_%s"%(DC,f), os.path.isfile(svg_folder+"I1_DC_%.05f_f_%s.npy"%(DC,f)))
   if not(os.path.isfile(svg_folder+"I1_DC_%.05f_f_%s.npy"%(DC,f))):
      pars = get_pars(DC, f)
      tf=100/pars["I1_f"]
      N=100000
      sample_rate = N/tf   

      ts=np.linspace(0, tf, N)
      I1 = get_I1(pars, ts)   

      X1_0 = get_stat(pars)
      X1s=odeint(dynamics, X1_0, ts, args=(pars,), hmax=0.001, atol=1e-7, rtol=1e-11, mxstep=5000)
      F = X1s.T[0]*I1
      np.save(svg_folder+"/I1_DC_%.05f_f_%s"%(DC,f), I1)
      np.save(svg_folder+"/F_DC_%.05f_f_%s"%(DC,f), F)
      np.save(svg_folder+"/t_DC_%.05f_f_%s"%(DC,f), ts)

def run_sims(svg_folder):
   DCs = np.linspace(0.0001, 0.002, 20)
   fs = np.logspace(-3, 2, 20)   
   np.save(svg_folder+"fs", fs)
   np.save(svg_folder+"DCs", DCs)
   g1, g2 = np.meshgrid(DCs, fs)
   p_vals = np.column_stack([g1.ravel(), g2.ravel()])
   
   Parallel(n_jobs=35)(delayed(sim)(p_vals[i,0], p_vals[i,1], svg_folder) for i in range(len(p_vals)))

def run_sims_f_var(DC, svg_folder):
   fs = np.logspace(-3, 2, 100)    
   np.save(svg_folder+"fs", fs)
   #Parallel(n_jobs=3)(delayed(sim)(DC, fs[i], svg_folder) for i in range(len(fs)))
   for i in range(len(fs)): sim(DC, fs[i], svg_folder)

   
#run_sims("data/diag/")

DC = 0.001
svg_folder = "data/1D/"
#run_sims_f_var(DC, svg_folder)

