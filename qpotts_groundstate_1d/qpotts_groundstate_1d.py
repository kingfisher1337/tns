import numpy as np
from numpy import tensordot as tdot
import potts
import gates
import ctm
import sys
import h0

q = int(sys.argv[1]) # number of Potts states
chi = int(sys.argv[2])
err = float(sys.argv[3])
verbose = np.any(map(lambda s: s == "--verbose", sys.argv))
tau = 1e-3
J = -1.0

if chi > 1:
    f = open("output/h_mz_q={:d}_J={:.0f}_chi={:d}_svalerr={:.0e}.dat".format(q, J, chi, err), "w")
    f2 = open("output/ctmrgsvals_q={:d}_J={:.0f}_chi={:d}_svalerr={:.0e}.dat".format(q, J, chi, err), "w")

mzops = [potts.magnetisationz_operator(q, j).reshape(q**2) for j in range(q)]

def do_point(h, dh, mz2, env2):
    if verbose:
        print h
    g = gates.imtime_evolution_from_pair_hamiltonian_mpo(
        potts.externalfield_term(q, -h),
        potts.interaction_term(q, J),
        tau)
    env, _, _, _ = ctm.ctmrg_square_1x1_invsymm(g, chi, env=env2, err=err, verbose=verbose)
    oenv = tdot(env.halfplane_upper(1), env.halfplane_lower(1), [[0,2],[2,0]])
    n = np.trace(oenv)
    oenv = oenv.reshape(q**2)
    mz = map(lambda op: np.abs(np.dot(oenv, op))/n, mzops)
    mzCmp = np.max(mz)

    while np.abs(mzCmp-mz2) > 1e-2 and dh > 1e-10:
        mz2,env2 = do_point(h - dh / 2.0, dh / 2.0, mz2, env2)
        dh = dh / 2.0
    
    f.write("{:e}".format(-h))
    for x in mz:
        f.write(" {:e}".format(x))
    f.write("\n")
    f.flush()
    
    s = env.rg_info["svals_hmove"]
    for x in (s/np.sqrt(np.dot(s,s))):
        f2.write("{:e} {:e}\n".format(-h, x))
        f2.flush()
    
    return mzCmp, env

env = None if chi == 1 else h0.get_h0_env(q, chi)
mz = 1.0
dh = 0.1
for h in np.arange(0, 2+dh, dh):
    mz, env = do_point(h, dh, mz, env)

"""
for h in np.linspace(0,2,41) if full else np.linspace(0.97, 1.02, 500):
    if verbose:
        print h
    h = -h
    
    g = gates.imtime_evolution_from_pair_hamiltonian_mpo(
        potts.externalfield_term(q, h),
        potts.interaction_term(q, J),
        tau)
    
    env, env2, curErr, it = ctm.ctmrg_square_1x1_invsymm(g, chi, env=env, err=err, verbose=verbose)

    if chi == 1:
        print "c =", env.c
        print "t1 =", env.t1
        print "t2 =", env.t2
        exit()

    oenv = tdot(env.halfplane_upper(1), env.halfplane_lower(1), [[0,2],[2,0]])
    n = np.trace(oenv)
    oenv = oenv.reshape(q**2)
    mz = map(lambda op: np.abs(np.dot(oenv, op))/n, mzops)
    
    f.write("{:e}".format(h))
    for x in mz:
        f.write(" {:e}".format(x))
    f.write("\n")
    f.flush()
    
    s = env.rg_info["svals_hmove"]
    for x in (s/np.sqrt(np.dot(s,s))):
        f2.write("{:e} {:e}\n".format(h, x))
        f2.flush()
"""

f.close()
f2.close()

