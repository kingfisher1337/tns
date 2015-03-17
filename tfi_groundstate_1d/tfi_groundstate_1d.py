import numpy as np
from numpy import tensordot as tdot
from numpy.linalg import svd
import gates
import ctm
import sys
import h0

sigmax_vec = gates.sigmax.reshape(4)
sigmaz_vec = gates.sigmaz.reshape(4)

def get_observable_env(e):
    return tdot(e.halfplane_upper(1), e.halfplane_lower(1), [[0,2],[2,0]])
    #return tdot(e.halfplane_left(1), e.halfplane_right(1), [[0,2],[2,0]])

def mylogspace(a, b, n):
    return (np.logspace(0, 1, n) - 1) * (b-a)/9.0 + a

def mylogspacereverse(a, b, n):
    return (b-mylogspace(a,b,n)+a)[::-1]

chi = int(sys.argv[1])
err = float(sys.argv[2])
tau = 1e-3
J = -1.0
env = h0.env.lift(chi)

f = open("output/h_sigmax_sigmaz_J={:.0f}_chi={:d}_svalerr={:.0e}.dat".format(J, chi, err), "w")

for h in np.concatenate([np.linspace(0,0.9,18,endpoint=False),np.linspace(0.9,1.1,50,endpoint=False),np.linspace(1.1,2,10)]):
#for h in np.concatenate([mylogspacereverse(0,1.05,30), mylogspace(1.05,2,5)[1:]]):
#for h in np.concatenate([mylogspacereverse(0,1.05,30), mylogspace(1.05,2,5)[1:]])[::-1]:
    h = -h

    ## start defining the imaginary time evolution iPEPO with the two-site part
    #g2 = gates.exp_sigmaz_sigmaz(-tau*J).reshape(4,4)
    #g1,s = svd(g2)[:2]
    #g1 = np.dot(g1, np.diag(np.sqrt(s))).reshape(2,2,4)
    ## note: the gate g has to be defined in a symmetric way!
    #g = np.einsum(g1, [0,4,1], g1.conj(), [4,2,3])
    ## add the sigmax-part:
    #g1 = gates.exp_sigmax(-0.5*tau*h)
    #g = np.einsum(g1, [0,4], g, [4,1,5,3], g1, [5,2])
    ## now we have a ready-to-use iPEPO defined by g
    
    g = gates.exp_sigmaz_sigmaz_mpo(-tau*J)
    g1 = gates.exp_sigmax(-0.5*tau*h)
    g = np.einsum(g1, [0,4], g, [4,1,5,3], g1, [5,2])
    
    #if env is None and chi > 16:
    #    env, env2, curErr, it = ctm.ctmrg_square_1x1_invsymm(g, 16, env=None, err=err, verbose=True)
    #    
    #    f2 = open("h0.dat", "w")
    #    for x in env.c.reshape(16*16):
    #        f2.write("{:e}, ".format(x))
    #    f2.write("\n")
    #    for x in env.t1.reshape(16*16*2):
    #        f2.write("{:e}, ".format(x))
    #    f2.write("\n")
    #    for x in env.t2.reshape(16*16*2):
    #        f2.write("{:e}, ".format(x))
    #    f2.write("\n")
    #    f2.close()
    #    exit()    
    #    env = env.lift(chi)

    env, env2, curErr, it = ctm.ctmrg_square_1x1_invsymm(g, chi, env=env, err=err, verbose=True)
    #for j in xrange(100):
    #    env, env2, curErr, it = ctm.ctmrg_square_1x1_invsymm(g, chi, env=env, err=err, max_iterations=1000)
    #    oenv = get_observable_env(env)
    #    n = np.trace(oenv)
    #    mz = np.abs(np.dot(oenv.reshape(4), sigmaz_vec))/n
    #    print "{:4d} {:e} {:e}".format(j, curErr, mz)
    #    if curErr < err:
    #        break
    
    oenv = get_observable_env(env)
    oenv2 = get_observable_env(env2)
    n = np.trace(oenv)
    n2 = np.trace(oenv2)
    oenv = oenv.reshape(4)
    oenv2 = oenv.reshape(4)
    mx = np.dot(oenv, sigmax_vec)/n
    mz = np.abs(np.dot(oenv, sigmaz_vec))/n
    mxerr = np.abs(np.dot(oenv2, sigmax_vec)/n2 - mx)
    mzerr = np.abs(np.abs(np.dot(oenv2, sigmaz_vec))/n2 - mz)
    print "{:e} {:e} {:e} {:e} {:e} {:e} {:d}".format(h, mx, mz, mxerr, mzerr, curErr, it)
    f.write("{:f} {:e} {:e} {:e} {:e} {:e} {:d}\n".format(h, mx, mz, mxerr, mzerr, curErr, it))
    f.flush()

f.close()

