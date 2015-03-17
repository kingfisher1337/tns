import numpy as np
import ctm
from ctm import ctmrg_square_1x1, ctmrg_square_2x2
from gates import sigmax, sigmaz
import sys

def correlator(e, a_1x2, z_1x2):
    n = np.tensordot(e, a_1x2, [[0,1,2,3,4,5], [0,1,2,3,4,5]])
    return np.tensordot(e, z_1x2, [[0,1,2,3,4,5], [0,1,2,3,4,5]]) / n

def err_fct(aDL, a_1x2, z_1x2):
    def err_fct_impl(e, e2):
        e = e.toarray_1x2(aDL,aDL)
        e2 = e2.toarray_1x2(aDL,aDL)
        return np.abs(correlator(e, a_1x2, z_1x2) - correlator(e2, a_1x2, z_1x2))
    return err_fct_impl

chi = int(sys.argv[1])
f = open("output/sigmazsigmaz_T_chi={:d}.dat".format(chi), "w")

env = ctm.CTMEnvironment2x2()
env1x1 = None

for T in np.linspace(0.0, 5.0, 31):
    print T
    theta = 0 if T == 0 else 0.5 * np.arcsin(np.exp(-1.0/T))

    alpha = np.array([[np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    a = np.fromfunction(lambda i,j,k,l,m: alpha[i,j]*alpha[i,k]*alpha[i,l]*alpha[i,m], (2,2,2,2,2), dtype=np.int64)
    aDL = np.einsum(a, [0,1,3,5,7], a, [0,2,4,6,8]).reshape(4,4,4,4)
    z = np.reshape(np.einsum(np.einsum(a, [5,1,2,3,4], sigmaz, [5,0]), [0,1,3,5,7], a, [0,2,4,6,8]), (4,4,4,4))
    a_1x2 = np.einsum(aDL, [0,6,4,5], aDL, [1,2,3,6])
    z_1x2 = np.einsum(z, [0,6,4,5], z, [1,2,3,6])
    
    testFct = lambda e, e2: np.abs(correlator(e.toarray_1x2(aDL,aDL), a_1x2, z_1x2) - correlator(e2.toarray_1x2(aDL,aDL), a_1x2, z_1x2))
    
    env1x1, err = ctmrg_square_1x1(aDL, chi, env=env1x1, err=1e-3)
    env1x1.to_2x2(env)
    
    env, err = ctmrg_square_2x2(aDL, aDL, chi, env=env, err=1e-3, convergenceTestFct=err_fct(aDL, a_1x2, z_1x2))
    e = env.toarray_1x2(aDL, aDL)
    
    #n = np.tensordot(e, a_1x2, [[0,1,2,3,4,5], [0,1,2,3,4,5]])
    #zz = np.tensordot(e, z_1x2, [[0,1,2,3,4,5], [0,1,2,3,4,5]]) / n
    zz = correlator(e, a_1x2, z_1x2)
    f.write("{:f} {:f} {:f}\n".format(T, zz, err))
    f.flush()

f.close()

