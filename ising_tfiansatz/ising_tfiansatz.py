import numpy as np
import ctm
from gates import sigmax, sigmaz

chi = 16

def mylogspace(a, b, n):
    return (np.logspace(0, 1, n) - 1) * (b-a)/9.0 + a
def mylogspacereverse(a, b, n):
    return (b-mylogspace(a,b,n)+a)[::-1]

f = open("output/J_h_T_E.dat", "w")
env = None

for T in np.concatenate([mylogspacereverse(0,2.2,25), mylogspace(2.2,5.0,25)[1:]]):
    theta = 0 if T == 0 else 0.5 * np.arcsin(np.exp(-1.0/T))
    alpha = np.array([[np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    a = np.fromfunction(lambda i,j,k,l,m: alpha[i,j]*alpha[i,k]*alpha[i,l]*alpha[i,m], (2,2,2,2,2), dtype=np.int64)
    aDL = np.einsum(a, [0,1,3,5,7], a, [0,2,4,6,8]).reshape(4,4,4,4)
    xDL = np.reshape(np.einsum(np.einsum(a, [5,1,2,3,4], sigmax, [5,0]), [0,1,3,5,7], a, [0,2,4,6,8]), (4,4,4,4))
    zDL = np.reshape(np.einsum(np.einsum(a, [5,1,2,3,4], sigmaz, [5,0]), [0,1,3,5,7], a, [0,2,4,6,8]), (4,4,4,4))
    aaDL = np.einsum(aDL, [0,6,4,5], aDL, [1,2,3,6]).reshape(4**6)
    axDL = np.einsum(aDL, [0,6,4,5], xDL, [1,2,3,6]).reshape(4**6)
    zzDL = np.einsum(zDL, [0,6,4,5], zDL, [1,2,3,6]).reshape(4**6)
    env, err = ctm.ctmrg_square_1x1_invsymm(aDL, chi, env=env)
    e = env.toarray1x2().reshape(4**6)
    n = np.dot(e, aaDL)
    mx = np.dot(e, axDL) / n
    czz = np.dot(e, zzDL) / n
    
    for J in [-1]:
        for h in [-10.0, -2.0, -1.0, -0.1]:
            E = 2*J*czz + h*mx
            f.write("{:f} {:f} {:f} {:f}\n".format(J, h, T, E))

f.close()

