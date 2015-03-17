import numpy as np
from gates import sigmax, sigmaz
import sys

f = open("output/sigmax_sigmaz_T_5x5.dat", "w")

for T in np.linspace(0.0, 5.0, 31):
    theta = 0 if T == 0 else 0.5 * np.arcsin(np.exp(-1.0/T))

    alpha = np.array([[np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    a = np.fromfunction(lambda i,j,k,l,m: alpha[i,j]*alpha[i,k]*alpha[i,l]*alpha[i,m], (2,2,2,2,2), dtype=np.int64)
    aDL = np.einsum(a, [0,1,3,5,7], a, [0,2,4,6,8]).reshape(4,4,4,4)
    
    x = np.reshape(np.einsum(np.einsum(a, [5,1,2,3,4], sigmax, [5,0]), [0,1,3,5,7], a, [0,2,4,6,8]), (4,4,4,4))
    z = np.reshape(np.einsum(np.einsum(a, [5,1,2,3,4], sigmaz, [5,0]), [0,1,3,5,7], a, [0,2,4,6,8]), (4,4,4,4))
    
    a2 = np.einsum(aDL, [0,1,6,4], aDL, [6,2,3,5]).reshape(4, 16, 4, 16)
    a4 = np.einsum(a2,  [0,1,6,4], a2,  [6,2,3,5]).reshape(4, 16*16, 4, 16*16)
    a5 = np.einsum(a4,  [0,1,6,4], aDL, [6,2,0,5]).reshape(16*16*4, 16*16*4)
    x5 = np.einsum(a4,  [0,1,6,4], x,   [6,2,0,5]).reshape(16*16*4, 16*16*4)
    z5 = np.einsum(a4,  [0,1,6,4], z,   [6,2,0,5]).reshape(16*16*4, 16*16*4)
    a5Pow4 = np.dot(a5, a5)
    a5Pow4 = np.dot(a5Pow4, a5Pow4)
    n = np.trace(np.dot(a5Pow4, a5))
    mx = np.trace(np.dot(a5Pow4, x5)) / n
    mz = np.trace(np.dot(a5Pow4, z5)) / n
    f.write("{:f} {:f} {:f}\n".format(T, mx, mz))

f.close()

