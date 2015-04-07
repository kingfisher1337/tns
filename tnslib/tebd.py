import numpy as np
from numpy import dot, einsum
from numpy import tensordot as tdot
from scipy.optimize import minimize
import ctm
import gates

def _make_double_layer_tensor(a, D):
    return einsum(a, [8,0,2,4,6], a.conj(), [8,1,3,5,7]).reshape([D**2]*4)

def _itebd_square_fu_singlebond(a, b, abg, env):
    tdot(b, b.conj(), [0,0])

def _itebd_square_cost_fct(env, abg, p, D):
    def cost_fct_impl(m):
        m = m.reshape(D**6, p**2)
        mH = m.conj().transpose()
        return dot(env, dot(m,mH).reshape(D**12)) - 2.0 * np.real(dot(env, dot(abg,mH).reshape(D**12)))
    return cost_fct_impl

def _itebd_square_fu_bond(p, D, a, b, g, env, err=1e-6, max_iterations=100):
    """
    env6---+---+-----+
     |     |   |     |
     |     0   1     |
     +--5         2--+
     |     4   3     |
     |     |   |     |
     +-----+---+-----+
    """
    idmat = np.identity(D, dtype=float)
    envVec = env.reshape(D**12)
    abg = einsum(a, [9,0,8,4,5], b, [10,1,2,3,8], g, [9,10,6,7])
    
    b2 = b
    
    d3 = None
    for j in xrange(max_iterations):
        d2 = None
        for j2 in xrange(max_iterations):
            S = tdot(
                    envVec, 
                    einsum(
                        tdot(abg, b2.conj(), [7,0]), [0,2,4,6,8,10,12,3,5,7,14], 
                        idmat, [1,13], 
                        idmat, [9,15], 
                        idmat, [11,16]
                    ).reshape(D**12,p,D,D,D,D), 
                    [0,0]
                ).reshape(p*D**4)
            R = einsum(
                tdot(_make_double_layer_tensor(b2,D), env, [[0,1,2],[1,2,3]]).reshape([D]*8), [2,7,1,6,3,8,4,9],
                idmat, [0,5]).reshape([p*D**4]*2)
            a2vec = np.linalg.lstsq(R, S)[0]
            a2 = a2vec.reshape(p,D,D,D,D)
            d = dot(a2vec.conj(), dot(R, a2vec)) - 2.0 * np.real(dot(a2vec.conj(), S))
            if d2 is not None and np.abs(d-d2) < err:
                break
            d2 = d
        
        d2 = None
        for j2 in xrange(max_iterations):
            S = tdot(
                envVec,
                einsum(
                    tdot(abg, a2.conj(), [6,0]), [0,2,4,6,8,10,12,1,16,9,11],
                    idmat, [3,13],
                    idmat, [5,14],
                    idmat, [7,15]
                ).reshape(D**12,p,D,D,D,D),
                [0,0]
            ).reshape(p*D**4)
            R = einsum(
                tdot(_make_double_layer_tensor(a2,D), env, [[0,2,3],[0,4,5]]).reshape([D]*8), [4,9,1,6,2,7,3,8],
                idmat, [0,5]).reshape([p*D**4]*2)
            b2vec = np.linalg.lstsq(R, S)[0]
            b2 = b2vec.reshape(p,D,D,D,D)
            d = dot(b2vec.conj(), dot(R, b2vec)) - 2.0 * np.real(dot(b2vec.conj(), S))
            if d2 is not None and np.abs(d-d2) < err:
                break
            d2 = d
    
        if d3 is not None and np.abs(d-d3) < err:
            break
        d3 = d
    
    return a2, b2

def itebd_square(a, b, gx, gy, chi, ctmrgerr=1e-6, ctmrg_max_iterations=1000000, tebd_max_iterations=1000000, tebd_update_err=1e-5, tebd_update_max_iterations=100, env=None):
    p, D = a.shape[:2]
    kronecker = np.fromfunction(np.vectorize(lambda j,k: 1. if j==k else 0), (p,p), dtype=int)
    
    gx2 = gx.swapaxes(0,1).swapaxes(2,3)
    gy2 = gy.swapaxes(0,1).swapaxes(2,3)
    
    aDL = _make_double_layer_tensor(a, D)
    bDL = _make_double_layer_tensor(b, D)
    
    mz = None
    
    for j in xrange(tebd_max_iterations):
    
        #if j % 100 == 0:
        print "[itebd_square] {:d} iterations done".format(j)
    
        env2 = env
        mz2 = mz
        print "[itebd_square] start ctmrg now"
        env, env2, err, num_iterations = ctm.ctmrg_square_2x2(aDL, bDL, chi, err=ctmrgerr, env=env2, iteration_bunch=10)
        xDL = einsum(einsum(a, [5,1,2,3,4], gates.sigmaz, [0,5]), [9,0,2,4,8], a.conj(), [9,1,3,5,7]).reshape(D**8)
        e = env.toarray1x1a(aDL, bDL).reshape(D**8)
        mz = dot(e, xDL) / dot(e, aDL.reshape(D**8))
        #if j % 10 == 0:
        if mz2 is not None:
            print "[itebd_square] mz estimate: {:.15e}; err: {:.15e}".format(mz, np.abs(mz-mz2))
            if np.abs(mz-mz2) < 1e-6:
                break
        
        a, b = _itebd_square_fu_bond(p, D, a, b, gx, env.toarray1x2ab(aDL,bDL))
        
        b, a = _itebd_square_fu_bond(p, D, np.rollaxis(b,1,5), np.rollaxis(a,1,5), gy2, np.rollaxis(env.toarray2x1ab(aDL,bDL),0,6))
        a, b = np.rollaxis(a,4,1), np.rollaxis(b,4,1)
        
        b, a = _itebd_square_fu_bond(p, D, b, a, gx2, env.toarray1x2ba(aDL,bDL))
        
        a, b = _itebd_square_fu_bond(p, D, np.rollaxis(a,1,5), np.rollaxis(b,1,5), gy, np.rollaxis(env.toarray2x1ba(aDL,bDL),0,6))
        a, b = np.rollaxis(a,4,1), np.rollaxis(b,4,1)
        """
        abgx = tdot(einsum(a, [6,0,8,4,5], b, [7,1,2,3,8]), gx, [[6,7],[0,1]])
        bagx = tdot(einsum(b, [6,0,8,4,5], a, [7,1,2,3,8]), gx, [[6,7],[0,1]])
        abgy = tdot(einsum(a, [7,8,2,3,4], b, [6,0,1,8,5]), gy, [[6,7],[0,1]])
        bagy = tdot(einsum(b, [7,8,2,3,4], a, [6,0,1,8,5]), gy, [[6,7],[0,1]])
        
        d2 = None
        e6 = env.toarray1x2ab(aDL, bDL)
        e12 = e6.reshape([D]*12)
        for k in xrange(tebd_update_max_iterations): # a-right b-left
            bDL2 = _make_double_layer_tensor(b, D)
            R = einsum(kronecker, [0,5], tdot(bDL2, e6, [[0,1,2],[1,2,3]]).swapaxes(0,1).reshape([D]*8), [1,6,2,7,3,8,4,9]).reshape([p*D**4]*2)
            S = einsum(e12, [5,1,6,11,7,12,8,13,9,3,10,4], tdot(abgx, b.conj(), [7,0]), [5,6,7,8,9,10,0,11,12,13,2]).reshape(p*D**4)
            a = np.linalg.lstsq(R, S)[0].reshape(p,D,D,D,D)
            aDL2 = _make_double_layer_tensor(a, D)
            R = einsum(kronecker, [0,5], tdot(e6, aDL2, [[0,4,5],[0,2,3]]).reshape([D]*8), [1,6,2,7,3,8,4,9]).reshape([p*D**4]*2)
            S = einsum(e12, [5,11,6,1,7,2,8,3,9,12,10,13], tdot(abgx, a.conj(), [6,0]), [5,6,7,8,9,10,0,11,4,12,13]).reshape(p*D**4)
            bvec = np.linalg.lstsq(R, S)[0]
            b = bvec.reshape(p,D,D,D,D)
            d = dot(bvec.conj(), dot(R, bvec)) - 2.0 * np.real(dot(bvec.conj(), S))
            if d2 is not None:
                if np.abs(d-d2) < tebd_update_err:
                    break
            d2 = d
        
        d2 = None
        e6 = env.toarray2x1ab(aDL, bDL)
        e12 = e6.reshape([D]*12)
        for k in xrange(tebd_update_max_iterations): # update a-up b-down
            bDL2 = _make_double_layer_tensor(b, D)
            R = einsum(kronecker, [0,5], tdot(bDL2, e6, [[0,1,3],[0,1,5]]).reshape([D]*8), [1,6,2,7,3,8,4,9]).reshape([p*D**4]*2)
            S = einsum(e12, [5,11,6,12,7,2,8,3,9,4,10,13], tdot(abgy, b.conj(), [6,0]), [5,6,7,8,9,10,0,11,12,1,13]).reshape(p*D**4)
            a = np.linalg.lstsq(R, S)[0].reshape(p,D,D,D,D)
            aDL2 = _make_double_layer_tensor(a, D)
            R = einsum(kronecker, [0,5], tdot(e6, aDL2, [[2,3,4],[1,2,3]]).swapaxes(2,3).reshape([D]*8), [1,6,2,7,3,8,4,9]).reshape([p*D**4]*2)
            S = einsum(e12, [5,1,6,2,7,11,8,12,9,13,10,4], tdot(abgy, a.conj(), [7,0]), [5,6,7,8,9,10,0,3,11,12,13]).reshape(p*D**4)
            bvec = np.linalg.lstsq(R, S)[0]
            b = bvec.reshape(p,D,D,D,D)
            d = dot(bvec.conj(), dot(R, bvec)) - 2.0 * np.real(dot(bvec.conj(), S))
            if d2 is not None:
                if np.abs(d-d2) < tebd_update_err:
                    break
            d2 = d
        
        d2 = None
        e6 = env.toarray1x2ba(aDL, bDL)
        e12 = e6.reshape([D]*12)
        for k in xrange(tebd_update_max_iterations): # b-right a-left
            bDL2 = _make_double_layer_tensor(b, D)
            R = einsum(kronecker, [0,5], tdot(e6, bDL2, [[0,4,5],[0,2,3]]).reshape([D]*8), [1,6,2,7,3,8,4,9]).reshape([p*D**4]*2)
            S = einsum(e12, [5,11,6,1,7,2,8,3,9,12,10,13], tdot(bagx, b.conj(), [6,0]), [5,6,7,8,9,10,0,11,4,12,13]).reshape(p*D**4)
            a = np.linalg.lstsq(R, S)[0].reshape(p,D,D,D,D)
            aDL2 = _make_double_layer_tensor(a, D)
            R = einsum(kronecker, [0,5], tdot(aDL2, e6, [[0,1,2],[1,2,3]]).swapaxes(0,1).reshape([D]*8), [1,6,2,7,3,8,4,9]).reshape([p*D**4]*2)
            S = einsum(e12, [5,1,6,11,7,12,8,13,9,3,10,4], tdot(bagx, a.conj(), [7,0]), [5,6,7,8,9,10,0,11,12,13,2]).reshape(p*D**4)
            bvec = np.linalg.lstsq(R, S)[0]
            b = bvec.reshape(p,D,D,D,D)
            d = dot(bvec.conj(), dot(R, bvec)) - 2.0 * np.real(dot(bvec.conj(), S))
            if d2 is not None:
                #print "[tebd_square] {:d} iterations done for b-right a-left; cost fct err: {:.15e}".format(k,np.abs(d-d2))
                if np.abs(d-d2) < tebd_update_err:
                    break
            d2 = d
        
        d2 = None
        e6 = env.toarray2x1ba(aDL, bDL)
        e12 = e6.reshape([D]*12)
        for k in xrange(tebd_update_max_iterations): # b-up a-down
            bDL2 = _make_double_layer_tensor(b, D)
            R = einsum(kronecker, [0,5], tdot(e6, bDL2, [[2,3,4],[1,2,3]]).reshape([D]*8).swapaxes(2,3), [1,6,2,7,3,8,4,9]).reshape([p*D**4]*2)
            S = einsum(e12, [5,1,6,2,7,11,8,12,9,13,10,4], tdot(bagy, b.conj(), [7,0]), [5,6,7,8,9,10,0,3,11,12,13]).reshape(p*D**4)
            a = np.linalg.lstsq(R, S)[0].reshape(p,D,D,D,D)
            aDL2 = _make_double_layer_tensor(a, D)
            R = einsum(kronecker, [0,5], tdot(aDL2, e6, [[0,1,3],[0,1,5]]).reshape([D]*8), [1,6,2,7,3,8,4,9]).reshape([p*D**4]*2)
            S = einsum(e12, [5,11,6,12,7,2,8,3,9,4,10,13], tdot(bagy, a.conj(), [6,0]), [5,6,7,8,9,10,0,11,12,1,13]).reshape(p*D**4)
            bvec = np.linalg.lstsq(R, S)[0]
            b = bvec.reshape(p,D,D,D,D)
            d = dot(bvec.conj(), dot(R, bvec)) - 2.0 * np.real(dot(bvec.conj(), S))
            if d2 is not None:
                #print "[tebd_square] {:d} iterations done for b-up a-down; cost fct err: {:.15e}".format(k,np.abs(d-d2))
                if np.abs(d-d2) < tebd_update_err:
                    break
            d2 = d
        """
        
        a /= np.max(np.abs(a))
        b /= np.max(np.abs(b))
        
        aDL = _make_double_layer_tensor(a, D)
        bDL = _make_double_layer_tensor(b, D)
        
    
    return a, b, env




def _itebd_square_pepo_invsymm_cost(R, S, D, kappa):
    def _itebd_square_pepo_invsymm_cost_impl(U):
        U = U.reshape(kappa*D, D)
        U2 = tdot(U, U, [1,1])
        return einsum(R, [0,1,2,3], U2, [0,1], U2, [2,3]) - 2.0 * einsum(S, [0,1], U2, [0,1])
    return _itebd_square_pepo_invsymm_cost_impl

def _init_downscaling_costfct(a, a2, D, kappa):
    def _init_downscaling_costfct_impl(x):
        U = x[:kappa*D**2].reshape(kappa*D, D)
        V = x[kappa*D**2:].reshape(kappa*D, D)
        aTest = einsum(a2, [0,5,6,7,8], V, [5,1], U, [6,2], V, [7,3], U, [8,4])
        return np.sum(np.abs(a - aTest))
    return _init_downscaling_costfct_impl

def itebd_square_pepo_invsymm(a, g, chi, env=None):
    if np.sum(np.abs(a - a.swapaxes(1,3))) > 1e-15:
        raise ValueError("given iPEPS is not invariant under spatial inversion")
    if np.sum(np.abs(a - a.swapaxes(2,4))) > 1e-15:
        raise ValueError("given iPEPS is not invariant under spatial inversion")
    if np.sum(np.abs(g - g.swapaxes(2,4))) > 1e-15:
        raise ValueError("given iPEPO is not invariant under spatial inversion")
    if np.sum(np.abs(g - g.swapaxes(3,5))) > 1e-15:
        raise ValueError("given iPEPO is not invariant under spatial inversion")

    p, D = a.shape[:2]
    kappa = g.shape[1]
    
    mz = None
    
    U2 = V2 = np.fromfunction(np.vectorize(lambda j,k: 1. if j==k else 0), (kappa*D,D), dtype=int)
    
    a2 = einsum(a, [9,1,3,5,7], g, [9,0,2,4,6,8]).reshape([p] + [kappa*D]*4)
    
    a3 = einsum(a2, [0,5,6,7,8], V2, [5,1], U2, [6,2], V2, [7,3], U2, [8,4])
    print a-a3
    print np.max(np.abs(a-a3))
    
    exit()
    x = minimize(_init_downscaling_costfct(a, a2, D, kappa), np.concatenate([U2.flatten(), V2.flatten()]))
    U2 = x.x[:kappa*D**2].reshape(kappa*D, D)
    V2 = x.x[kappa*D**2:].reshape(kappa*D, D)
    
    print x
    #print U
    #print V
    exit()
    
    for j in xrange(5):
        aDL = _make_double_layer_tensor(a, D)
        env, env2, err, num_iterations = ctm.ctmrg_square_1x1_invsymm(aDL, chi, env=env, verbose=True)
        
        xDL = einsum(einsum(a, [5,1,2,3,4], gates.sigmaz, [0,5]), [9,0,2,4,8], a.conj(), [9,1,3,5,7]).reshape(D**8)
        e = env.toarray1x1().reshape(D**8)
        mz, mz2 = dot(e, xDL) / dot(e, aDL.reshape(D**8)), mz
        if mz2 is not None:
            print "[itebd_square_pepo_invsymm] mz estimate: {:.15e}; err: {:.15e}".format(mz, np.abs(mz-mz2))
            if np.abs(mz-mz2) < 1e-6:
                break

        a2 = einsum(a, [9,1,3,5,7], g, [9,0,2,4,6,8]).reshape([p] + [kappa*D]*4)
        
        e = env.toarray1x2()
        a2L = einsum(a2, [0,5,2,6,4], V2, [5,1], V2, [6,3])
        a2R = einsum(a2L, [0,1,5,3,4], U2, [5,2]).reshape(p,D,kappa*D,D,D)
        a2L = einsum(a2L, [0,1,2,3,5], U2, [5,4]).reshape(p,D,D,D,kappa*D)
        a2L = einsum(a2L, [8,0,2,4,6], a2L, [8,1,3,5,7]).reshape(D**2,(kappa*D)**2,D**2,D**2)
        a2R = einsum(a2R, [8,0,2,4,6], a2R, [8,1,3,5,7]).reshape(D**2,D**2,D**2,(kappa*D)**2)
        R = einsum(einsum(e, [4,1,2,3,5,6], a2L, [4,0,5,6]), [0,2,3,4], a2R, [2,3,4,1]).reshape([kappa*D]*4).swapaxes(1,2)
        S = einsum(R, [2,2,0,1])
        U = minimize(_itebd_square_pepo_invsymm_cost(R,S,D,kappa), U2.reshape(kappa*D**2)).x.reshape(kappa*D, D)
        #print U
        
        e = env.toarray2x1()
        a2U = einsum(a2, [0,1,5,3,6], U2, [5,2], U2, [6,4])
        a2D = einsum(a2U, [0,1,2,5,4], V2, [5,3])
        a2U = einsum(a2U, [0,5,2,3,4], V2, [5,1])
        a2U = einsum(a2U, [8,0,2,4,6], a2U, [8,1,3,5,7]).reshape(D**2,D**2,(kappa*D)**2,D**2)
        a2D = einsum(a2D, [8,0,2,4,6], a2D, [8,1,3,5,7]).reshape((kappa*D)**2,D**2,D**2,D**2)
        R = einsum(e, [2,3,4,5,6,7], a2U, [2,3,0,7], a2D, [1,4,5,6]).reshape([kappa*D]*4).swapaxes(1,2)
        S = einsum(R, [2,2,0,1])
        V = minimize(_itebd_square_pepo_invsymm_cost(R,S,D,kappa), V2.reshape(kappa*D**2)).x.reshape(kappa*D, D)
        #print V
        
        a = einsum(a2, [0,5,6,7,8], V, [5,1], U, [6,2], V, [7,3], U, [8,4])
        U2,V2 = U,V
        
        print a
    
    return a, env, j+1


