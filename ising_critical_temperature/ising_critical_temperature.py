import numpy as np
import sys

Lx = 20
Ly = 6
f = open("output/entropy_bulk_T.dat", "w")
f2 = open("output/entropy_Lx=2_T.dat", "w")

for T in np.linspace(0, 6, 7+6*4):
    if T == 0:
        theta = 0
    else :
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

    svals = np.linalg.svd(rho, compute_uv=False)
    svals = svals[np.nonzero(svals)]
    svals /= np.sqrt(np.dot(svals, svals))
    S = -np.sum(2.0 * svals**2 * np.log(svals))
    f2.write("{:f} {:f}\n".format(T, S))

    for l in range(Lx):
        rho = np.einsum(rho, [0,1], aRing, [2,3,0,1])

    svals = np.linalg.svd(rho, compute_uv=False)
    svals = svals[np.nonzero(svals)]
    rho = aRing = None # free memory
    svals /= np.sqrt(np.dot(svals, svals))
    S = -np.sum(svals * np.log(svals))
    f.write("{:f} {:f}\n".format(T, S))

f.close()
f2.close()

