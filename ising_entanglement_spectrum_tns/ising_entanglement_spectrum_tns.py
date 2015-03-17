import numpy as np
import sys
from symmetries import *

Lx = 10
Ly = 6
T = float(sys.argv[1])

theta = 0.5 * np.arcsin(np.exp(-1.0/T))
alpha = np.array([[np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]])

if Ly == 6:
    r = np.fromfunction(lambda i,j,k,l: alpha[i,j]*alpha[i,k]*alpha[i,l], (2,2,2,2), dtype=np.int64)
    r1 = np.reshape(np.einsum(r, [0,1,3,5], r, [0,2,4,6]), (4,2**1,2**1,4))
    r2 = np.reshape(np.einsum(r1, [1,2,4,0], r1, [0,3,5,6]), (4,2**2,2**2,4))
    r4 = np.reshape(np.einsum(r2, [1,2,4,0], r2, [0,3,5,6]), (4,2**4,2**4,4))
    rho = np.reshape(np.einsum(r4, [1,2,4,0], r2, [0,3,5,1]), tuple([2**Ly]*2))
    r = r1 = r2 = r4 = None # free memory

    a = np.fromfunction(lambda i,j,k,l,m: alpha[i,j]*alpha[i,k]*alpha[i,l]*alpha[i,m], (2,2,2,2,2), dtype=np.int64)
    a1 = np.reshape(np.einsum(a, [0,1,3,5,7], a, [0,2,4,6,8]), (4,2,2,4,2,2))
    a2 = np.reshape(np.einsum(a1, [1,2,4,0,7,9], a1, [0,3,5,6,8,10]), (4,4,4,4,4,4))
    a4 = np.reshape(np.einsum(a2, [1,2,4,0,7,9], a2, [0,3,5,6,8,10]), (4,2**4,2**4,4,2**4,2**4))
    aRing = np.reshape(np.einsum(a4, [1,2,4,0,7,9], a2, [0,3,5,1,8,10]), tuple([2**Ly]*4))
    a = a1 = a2 = a4 = None # free memory
else:
    print "only implemented for Ly=6!"
    exit(-1)

S = Symmetries([translationSymmetry, spinFlipSymmetry], Ly)

entropyList = []

for l in range(1,Lx+1):
    info, B = S.blockDiagonalise(rho)
    k = map(lambda x: x[0], info)
    s = map(lambda x: x[1], info)
    
    svals = []
    norm = 0
    for Bj in B:
        svals.append(np.linalg.svd(Bj, compute_uv=False))
        norm += np.dot(svals[-1], svals[-1])
    entropy = 0
    
    f = open("output/entanglement_spectrum_T={:.2f}_Ly={:d}_Lx={:d}.dat".format(T, Ly, 2*l), "w")
    
    for j in range(len(B)):
        svals[j] /= np.sqrt(norm)
        entropy -= np.sum(2.0 * svals[j]**2 * np.log(svals[j]))
        kval = k[j]
        if kval > np.pi + 1e-2:
            kval -= 2.0 * np.pi
        for xi in svals[j]:
            f.write("{:f} {:f} {:f}\n".format(kval, s[j], xi))
        f.write("\n")
    f.close()
    
    entropyList.append(entropy)
    
    rho = np.einsum(rho, [0,1], aRing, [2,3,0,1])

f = open("output/entropy_l_Ly={:d}_T={:.2f}.dat".format(Ly, T), "w")
for l in range(1, Lx+1):
    f.write("{:d} {:f}\n".format(l, entropyList[l-1]))
f.close()


