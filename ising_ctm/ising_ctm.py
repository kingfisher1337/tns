import numpy as np
import ctm
from gates import sigmax, sigmaz
import sys

def eval_site(e, x):
    return np.tensordot(e, x, [[0,1,2,3], [0,1,2,3]])
def eval_sitepair(e, x):
    return np.tensordot(e, x, [[0,1,2,3,4,5], [0,1,2,3,4,5]])

def err_fct(a, x, z):
    def err_fct_impl(e, e2):
        e = e.toarray1x1()
        e2 = e2.toarray1x1()
        n = eval_site(e, a)
        n2 = eval_site(e2, a)
        return (np.abs(eval_site(e,x)/n - eval_site(e2,x)/n2) + np.abs(eval_site(e,z)/n - eval_site(e2,z)/n2))
    return err_fct_impl

def mylogspace(a, b, n):
    return (np.logspace(0, 1, n) - 1) * (b-a)/9.0 + a

def mylogspacereverse(a, b, n):
    return (b-mylogspace(a,b,n)+a)[::-1]

chi = int(sys.argv[1])
f = open("output/sigmax_sigmaz_T_chi={:d}.dat".format(chi), "w")
f2 = open("output/sigmazsigmaz_T_chi={:d}.dat".format(chi), "w")

env = None

#for T in np.linspace(0.0, 5.0, 31):
for T in np.concatenate([mylogspacereverse(0,2.2,25), mylogspace(2.2,5.0,25)[1:]]):
    print T
    theta = 0 if T == 0 else 0.5 * np.arcsin(np.exp(-1.0/T))

    alpha = np.array([[np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    a = np.fromfunction(lambda i,j,k,l,m: alpha[i,j]*alpha[i,k]*alpha[i,l]*alpha[i,m], (2,2,2,2,2), dtype=np.int64)
    aDL = np.einsum(a, [0,1,3,5,7], a, [0,2,4,6,8]).reshape(4,4,4,4)
    x = np.reshape(np.einsum(np.einsum(a, [5,1,2,3,4], sigmax, [5,0]), [0,1,3,5,7], a, [0,2,4,6,8]), (4,4,4,4))
    z = np.reshape(np.einsum(np.einsum(a, [5,1,2,3,4], sigmaz, [5,0]), [0,1,3,5,7], a, [0,2,4,6,8]), (4,4,4,4))
    
    #env, err = ctm.ctmrg_square_1x1(aDL, chi, env=env, convergenceTestFct=err_fct(aDL,x,z))
    #env, err = ctm.ctmrg_square_1x1(aDL, chi, env=env, step_method="directional")
    #env, err = ctm.ctmrg_square_1x1(aDL, chi, env=env, step_method="radial")
    env, err = ctm.ctmrg_square_1x1_invsymm(aDL, chi, env=env)
    
    e = env.toarray1x1()
    n = eval_site(e, aDL)
    mx = np.abs(eval_site(e, x)) / n
    mz = np.abs(eval_site(e, z)) / n
    f.write("{:f} {:f} {:f}\n".format(T, mx, mz))
    
    e = env.toarray1x2()
    aa = np.einsum(aDL, [0,6,4,5], aDL, [1,2,3,6])
    zz = np.einsum(z, [0,6,4,5], z, [1,2,3,6])
    n = eval_sitepair(e, aa)
    czz = eval_sitepair(e, zz) / n
    f2.write("{:f} {:f}\n".format(T, czz))

f.close()
f2.close()

