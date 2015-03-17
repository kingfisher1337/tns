import numpy as np
import sys
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from symmetries import *

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = "\usepackage{bm}"


if len(sys.argv) < 4:
    print "usage: python ising_entanglement_spectrum.py Lx Ly T"
    exit(-1)
Lx = int(sys.argv[1])
Ly = int(sys.argv[2])
T = float(sys.argv[3])
beta = 1.0 / T
p = 2
vdim = p**Ly

if len(sys.argv) >= 6:
    ymin = float(sys.argv[4])
    ymax = float(sys.argv[5])
else:
    ymin = None
    ymax = None

sys.stderr.write("started Lx=" + str(Lx) + ", Ly=" + str(Ly) + ", T=" + str(1.0/beta) + "\n")
t0 = time.time()

spin = np.ndarray((vdim, Ly))
for i in range(vdim):
    for j in range(Ly):
        spin[i,j] = 2*((i>>j)&1)-1

# note: use H(sigma_1, ..., sigma_N) = sum_<i,j> (sigma_i sigma_j + 1) to avoid overflows!
Fv = np.fromfunction(lambda i,k: np.exp(beta*(spin[i,k]*spin[i,(k+1)%Ly])), (vdim,Ly), dtype=np.int64).prod(1)
Fh = np.fromfunction(lambda i,j,k: np.exp(beta*(spin[i,k]*spin[j,k])), (vdim,vdim,Ly), dtype=np.int64).prod(2)
M = np.fromfunction(lambda i,j: Fv[i]*Fh[j,i], (vdim,vdim), dtype=np.int64)
G = Fv

S = Symmetries([translationSymmetry, spinFlipSymmetry], Ly)
printCommutators = False
if printCommutators:
    Top = S.getSymmetryOperationAsMatrix(0)
    Sop = S.getSymmetryOperationAsMatrix(1)

Lx = Lx / 2
entropyList = []
for l in range(1, Lx+1):
    A = np.fromfunction(lambda i,j: np.sqrt(Fh[i,j]*G[i]*G[j]), (vdim,vdim), dtype=np.int64)
    
    info, B = S.blockDiagonalise(A)
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
        entropy -= np.sum(2.0 * svals[j]**2 * np.log(svals[j]))
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
    if not ymin is None:
        plt.ylim(ymin, ymax)
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
        sys.stderr.write("T-commutator (l=" + str(l) + "): " + str(np.sum(np.abs(np.dot(A, Top) - np.dot(Top, A)))) + "\n")
        sys.stderr.write("S-commutator (l=" + str(l) + "): " + str(np.sum(np.abs(np.dot(A, Sop) - np.dot(Sop, A)))) + "\n")
    
    if l+1 < Lx:
        G = np.dot(M, G)

sys.stderr.write(str(time.time() - t0) + "seconds runtime\n")

#plt.clf()
#plt.cla()
plt.plot(range(1, Lx+1), entropyList, marker="x")
#plt.yscale('log')
plt.xlabel("$l = \\frac{L_x}{2}$")
plt.ylabel("$S = \\sum\\limits_j \\xi_j \\ln(\\xi_j)$")
plt.grid(True)
plt.savefig("plots/entropy_l_Ly=" + str(Ly) + "_T=" + "{:.2f}".format(T) + ".png")


