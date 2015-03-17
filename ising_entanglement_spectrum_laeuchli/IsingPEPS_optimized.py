#!/usr/bin/python
#//Library/Frameworks/Python.framework/Versions/Current/bin/python

from __future__ import division

from sys import *
import math
import string
import numpy as np
from scipy import linalg
import random
import time
import copy
#import matplotlib.pyplot as plt
#import ManyBodyHilbert as mbh

# MIRMOD BEGIN
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from symmetries import *

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = "\usepackage{bm}"
# MIRMOD END

#import mayavi.mlab as mlab
#from matplotlib.backends.backend_pdf import PdfPages

timing=1

L=int(argv[1])
W=int(argv[2])
T=float(argv[3])
beta=1./T
cdim = 2**W;

print "### Ising PEPS Python tool ###"
print "### (C) AML 2014    ###"
print "SVDing a system with depth",L,", width W =",W," and T=",T

"""MIRMOD
def spin(conf,pos):
    return 2*((conf>>pos)&1)-1
"""
# MIRMOD BEGIN
spin = np.ndarray((cdim, W))
for i in range(cdim):
    for j in range(W):
        spin[i,j] = 2*((i>>j)&1)-1
# MIRMOD END

def NestedStripZ(SZ,W,beta):
    t1=time.time()
    cdim=2**W;
    ret=np.zeros((cdim));
    ergpart1 = np.fromfunction(lambda i,p: -spin[i,p]*spin[i,(p+1)%W], (cdim,W), dtype=np.int64).sum(1)
    zline1=np.exp(-beta*(ergpart1+W))        
    zline2 = np.ones((cdim))
    pm=np.outer(zline2,zline1)
    
    ergmat=np.zeros((cdim,cdim))
    for p in range(W):
        spinline=np.zeros((cdim))
        for i in range(cdim):
            spinline[i]=spin[i,p]
        ergmat-=np.outer(spinline,spinline)
    ergmat+=W
    pm*=np.exp(-beta*(ergmat))

    ret=np.dot(pm,SZ)
    ret=np.sqrt(ret)
    if timing:
        print "NestedStripZ took",time.time()-t1,"seconds"
    return ret

def PsiMat(SZ,W,beta):
    t1 = time.time()
    cdim = 2**W;
    pm=np.zeros((cdim,cdim))
    zline=np.zeros((cdim))
    for i in range(cdim):
        ergline = np.fromfunction(lambda p: -spin[i,p]*spin[i,(p+1)%W], (W,), dtype=np.int64).sum()
        zline[i]=math.exp(-beta/2*(ergline+W))
    pm=np.outer(zline,zline)

    ergmat=np.zeros((cdim,cdim))
    for p in range(W):
        spinline=np.zeros((cdim))
        for i in range(cdim):
            spinline[i]=spin[i,p]
        ergmat+=np.outer(spinline,spinline)
    ergmat+=W
    pm*=np.exp(-beta/2*(ergmat))
    pm*=np.outer(SZ,SZ)

    if timing:
        print "PsiMat took",time.time()-t1,"seconds"
    return pm

# MIRMOD BEGIN
Ly = W
S = Symmetries([translationSymmetry, spinFlipSymmetry], Ly)
printCommutators = True
if printCommutators:
    Top = S.getSymmetryOperationAsMatrix(0)
    Sop = S.getSymmetryOperationAsMatrix(1)
entropyList = []
# MIRMOD END

SZ=np.zeros((cdim))+1.
for l in range(L):
    pm=PsiMat(SZ,W,beta)
    t1=time.time()
    #U,svl,Vh = linalg.svd(pm)
    
    # MIRMOD BEGIN
    info, B = S.blockDiagonalise(pm)
    k = map(lambda x: x[0], info)
    s = map(lambda x: x[1], info)
    
    svals = []
    norm = 0
    for Bj in B:
        svals.append(np.linalg.svd(Bj, compute_uv=False))
        norm += np.dot(svals[-1], svals[-1])
    entropy = 0
    plt.clf()
    for j in range(len(B)):
        svals[j] /= np.sqrt(norm)
        entropy -= np.sum(svals[j] * np.log(svals[j]))
        kval = k[j]
        if kval > np.pi + 1e-2:
            kval -= 2.0 * np.pi
        symbol = "r+" if s[j] == 0 else "bx"
        plt.plot([kval] * len(svals[j]), -2.0*np.log(svals[j]), symbol)
    plt.title("$T = " + str(T) + "$, $H = 0$,\n$L_y = " + str(Ly) + "$, $L_x = " + str(2*l) + "$, $\\ell = " + str(l) + "$,\n$S = " + str(entropy) + "$")
    plt.subplots_adjust(top=0.85)
    plt.axes().set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    plt.axes().set_xticklabels(["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])
    plt.xlim(-np.pi-0.2, np.pi+0.2)
    plt.ylim(ymin=-2)
    plt.xlabel("$k$")
    plt.ylabel("$\\xi \\quad$ (note: $\\xi^2 \\in \\sigma(\\rho_\\ell)$, even $\\hat{=}$ red, odd $\\hat{=}$ blue)")
    plt.grid(True)
    ylim = plt.axes().get_ylim()
    aspectRatio = 3.0 * (2.0*np.pi+0.4) / (ylim[1] - ylim[0])
    plt.axes().set_aspect(aspectRatio)
    plt.gcf().set_size_inches(4.0, 6.0)
    plt.savefig("plots/entanglement_spectrum_T=" + "{:.2f}".format(T) + "_Ly=" + str(Ly) + "_Lx=" + str(2*l) + ".png", dpi=300)
    plt.close()
    
    entropyList.append(entropy)
    
    if printCommutators:
        A = pm
        sys.stderr.write("T-commutator (l=" + str(l) + "): " + str(np.sum(np.abs(np.dot(A, Top) - np.dot(Top, A)))) + "\n")
        sys.stderr.write("S-commutator (l=" + str(l) + "): " + str(np.sum(np.abs(np.dot(A, Sop) - np.dot(Sop, A)))) + "\n")
    # MIRMOD END
    
    """MIRMOD
    svl = linalg.svd(pm,compute_uv=False)
    if timing:
        print "SVD took",time.time()-t1,"seconds"
    nrm=np.dot(svl,svl)
    svl/=math.sqrt(nrm)
    entr=0
    for i in range(len(svl)):
        entr+= -svl[i]**2*2.*math.log(svl[i])
    print "l=",l+1," vN Entropy = ",entr
    print "xi0",-math.log(svl[0])
    for i in range(len(svl)):
        #print "xi",i,-math.log(svl[i])
        print "dxi",i,-math.log(svl[i])+math.log(svl[0])
    """
    if(l<L-1):
        SZold = copy.deepcopy(SZ)
        SZ=NestedStripZ(SZold,W,beta)
        
# MIRMOD BEGIN
Lx = L
plt.plot(range(1, Lx+1), entropyList)
plt.xlabel("$l = \\frac{L_x}{2}$")
plt.ylabel("$S = \\sum\\limits_j \\xi_j \\ln(\\xi_j)$")
plt.grid(True)
plt.savefig("plots/entropy_l_Ly=" + str(Ly) + "_T=" + "{:.2f}".format(T) + ".png")
# MIRMOD END

