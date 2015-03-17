import numpy as np

def _trg_square_svd(a, D, Dcut, transpose=False):
    if transpose:
        a = a.swapaxes(1,2).swapaxes(1,3)
    u,s,v = np.linalg.svd(a.reshape(D*D, D*D), full_matrices=False)
    s = s[:Dcut] 
    s /= np.sqrt(np.dot(s, s))
    smat = np.diag(np.sqrt(s))
    u = np.dot(u[:,:Dcut], smat).reshape(D, D, Dcut)
    v = np.dot(smat, v[:Dcut]).reshape(Dcut, D, D)
    return u, v, s

def _trg_square_contract(a, b, c, d):
    t1 = np.tensordot(a, b, [1,2])
    t2 = np.tensordot(c, d, [1,1])
    return np.tensordot(t1, t2, [[1,3],[2,0]])
    #t1 = np.einsum(a, [0,4,2], b, [1,3,4])
    #t2 = np.einsum(c, [2,4,0], d, [3,4,1])
    #return np.einsum(t1, [0,1,4,5], t2, [2,3,5,4])

def trg_square(chi, a, b = None, o1 = None, o2 = None, o3 = None, o4 = None, maxIterations = 10000, err = 1e-8):
    """
       |   |   |   |   |   |
    ---b---a---b---a---b---a---
       |   |   |   |   |   |
    ---a---b---a---b---a---b---
       |   |   |   |   |   |
    ---b---a---o1--o2--b---a---
       |   |   |   |   |   |
    ---a---b---o3--o4--a---b---
       |   |   |   |   |   |
    ---b---a---b---a---b---a---
       |   |   |   |   |   |
    ---a---b---a---b---a---b---
       |   |   |   |   |   |
    """
    
    if b is None:
        b = a
    if o1 is None:
        o1 = b
    if o2 is None:
        o2 = a
    if o3 is None:
        o3 = a
    if o4 is None:
        o4 = b
    
    sPrev = np.zeros(4*chi)
    
    for j in range(maxIterations):
        D = a.shape[0]
        Dcut = np.min([D*D, chi])
        
        ua, va, sa = _trg_square_svd(a,  D, Dcut, True)
        ub, vb, sb = _trg_square_svd(b,  D, Dcut)
        u1, v1, s1 = _trg_square_svd(o1, D, Dcut)
        u2, v2, s2 = _trg_square_svd(o2, D, Dcut, True)
        u3, v3, s3 = _trg_square_svd(o3, D, Dcut, True)
        u4, v4, s4 = _trg_square_svd(o4, D, Dcut)
        
        o1 = _trg_square_contract(va, v1, u3, ub)
        o2 = _trg_square_contract(va, vb, u2, u1)
        o3 = _trg_square_contract(v3, v4, ua, ub)
        o4 = _trg_square_contract(v2, vb, ua, u4)
        a = b = _trg_square_contract(va, vb, ua, ub)
        
        if Dcut == chi:
            sVec = np.concatenate((s1,s2,s3,s4))
            if np.all(np.abs(sVec - sPrev) < err):
                print "[trg] needed", j, "iterations"
                break
            sPrev = sVec
    
    t1 = np.tensordot(o1, o3, [[2,0],[0,2]])
    t2 = np.tensordot(o2, o4, [[2,0],[0,2]])
    return np.tensordot(t1, t2, [[0,2,1,3],[1,3,0,2]])
    #t1 = np.einsum(o1, [5,0,4,2], o3, [4,1,5,3])
    #t2 = np.einsum(o2, [5,0,4,2], o4, [4,1,5,3])
    #t1 = np.einsum(t1, [4,5,0,1], t2, [2,3,4,5])
    #return np.einsum(t1, [0,1,0,1])

