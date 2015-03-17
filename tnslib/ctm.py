import numpy as np
from numpy import dot
from numpy import tensordot as tdot
from numpy.linalg import svd
from copy import copy

def _ctm_array_dist(a, b):
    return np.sum(np.abs(a.toarray() - b.toarray()))

class CTMEnvironment2x2:
    """
    C1----T1b----T1a----C2
    |      |      |      |
    T4b--            --T2a
    |                    |
    T4a--            --T2b
    |      |      |      |
    C4----T3a----T3b----C3
    """
    def __init__(self, D=1, chi=1, dtype=float, init=True):
        self.c1 = np.ones((chi, chi), dtype=dtype)
        self.c2 = np.ones((chi, chi), dtype=dtype)
        self.c3 = np.ones((chi, chi), dtype=dtype)
        self.c4 = np.ones((chi, chi), dtype=dtype)
        self.t1a = np.ones((chi, D, chi), dtype=dtype)
        self.t1b = np.ones((chi, D, chi), dtype=dtype)
        self.t2a = np.ones((chi, D, chi), dtype=dtype)
        self.t2b = np.ones((chi, D, chi), dtype=dtype)
        self.t3a = np.ones((chi, D, chi), dtype=dtype)
        self.t3b = np.ones((chi, D, chi), dtype=dtype)
        self.t4a = np.ones((chi, D, chi), dtype=dtype)
        self.t4b = np.ones((chi, D, chi), dtype=dtype)
    def toarray(self):
        t1 = tdot(self.c1, tdot(self.t1b, self.t1a, [2,0]), [1,0])
        t2 = tdot(self.c2, tdot(self.t2a, self.t2b, [2,0]), [1,0])
        t3 = tdot(self.c3, tdot(self.t3b, self.t3a, [2,0]), [1,0])
        t4 = tdot(self.c4, tdot(self.t4a, self.t4b, [2,0]), [1,0])
        t1 = tdot(t1, t2, [3,0])
        t3 = tdot(t3, t4, [3,0])
        return tdot(t1, t3, [[5,0],[0,5]])
        #tdot(tdot(tdot(tdot(tdot(tdot(tdot(tdot(tdot(tdot(tdot(tdot(self.c1, self.t1b, [1,0]), t1a, [2,0]), c2, [3,0]), t2a, [3,0]), t2b, [4,0]), t2a, [5,0]), c3, [6,0]), t3b, [6,0]), t3a, [7,0]), c4, [8,0]), t4a, [8,0]), t4b, [[0,8], [2,0]])
    def toarray_1x2(self, a, b):
        """
        C1----T1b----T1a----C2
        |      |      |      |
        T4b----a------b----T2a
        |      |      |      |
        T4a--            --T2b
        |      |      |      |
        C4----T3a----T3b----C3
        """
        e = self.toarray()
        return tdot(a, tdot(b, e, [[0,1],[1,2]]), [[0,1,3],[2,1,7]])

def _ctmrg_square_2x2_step_singlecol(c1, c4, t1, t3, t4a, t4b, a, b, D, chi):
    """
    left move for
    C1----T1----?
    |      |
    T4b----a----?
    |      |
    T4a----b----?
    |      |
    C4----T3----?
    """
    c1 = tdot(c1, t1, [1,0]).reshape(chi*D, chi)
    c4 = tdot(t3, c4, [2,0]).swapaxes(1,2).reshape(chi, chi*D)
    t4a2 = tdot(t4b, a, [1,3]).swapaxes(1,4).swapaxes(3,4).swapaxes(2,4).reshape(chi*D, D, chi*D)
    t4b2 = tdot(t4a, a, [1,3]).swapaxes(1,4).swapaxes(3,4).swapaxes(2,4).reshape(chi*D, D, chi*D)
    q1 = tdot(t4a2, c1, [2,0]).reshape(chi*D, chi*D)
    q4 = tdot(c4, t4b2, [1,0]).reshape(chi*D, chi*D)
    
    v,z = np.linalg.eigh(dot(c1, c1.conj().transpose()) + dot(c4.conj().transpose(), c4))
    idx = v.argsort()[::-1]
    z = z[:,idx][:,:chi]
    
    c1 = dot(z.conj().transpose(), c1)
    c4 = dot(c4, z)
    
    v,w = np.linalg.eigh(dot(q1, q1.conj().transpose()) + dot(q4.conj().transpose(), q4))
    idx = v.argsort()[::-1]
    w = w[:,idx][:,:chi]
    
    t4a = tdot(tdot(w.conj(), t4a2, [0,0]), z, [2,0])
    t4b = tdot(tdot(z.conj(), t4b2, [0,0]), w, [2,0])
    
    return c1, c4, t4a, t4b
    
def _ctmrg_square_2x2_step(c1, c4, t1a, t1b, t3a, t3b, t4a, t4b, a, b, D, chi):
    c1, c4, t4a, t4b = _ctmrg_square_2x2_step_singlecol(c1, c4, t1b, t3a, t4a, t4b, a, b, D, chi)
    c1, c4, t4a, t4b = _ctmrg_square_2x2_step_singlecol(c1, c4, t1a, t3b, t4b, t4a, b, a, D, chi)
    c1 /= np.max(np.abs(c1))
    c4 /= np.max(np.abs(c4))
    t4a /= np.max(np.abs(t4a))
    t4b /= np.max(np.abs(t4b))
    return c1, c4, t4a, t4b

def ctmrg_square_2x2(a, b, chi, err=1e-6, maxIterations=100000, iterationBunch=100, env=None, convergenceTestFct=_ctm_array_dist):
    D = a.shape[0]
    if not isinstance(env, CTMEnvironment2x2):
        env = CTMEnvironment2x2(D, chi, type(a[0,0,0,0]))
    a2 = a.swapaxes(0,1).swapaxes(1,2).swapaxes(2,3)
    a3 = a.swapaxes(0,2).swapaxes(1,3)
    a4 = a2.swapaxes(0,2).swapaxes(1,3)
    b2 = b.swapaxes(0,1).swapaxes(1,2).swapaxes(2,3)
    b3 = b.swapaxes(0,2).swapaxes(1,3)
    b4 = b2.swapaxes(0,2).swapaxes(1,3)
    
    env2 = None
    curErr = None
    
    for j in xrange(maxIterations/iterationBunch):
        for j2 in xrange(iterationBunch):
            env.c1, env.c4, env.t4a, env.t4b = _ctmrg_square_2x2_step(env.c1, env.c4, env.t1a, env.t1b, env.t3a, env.t3b, env.t4a, env.t4b, a, b, D, chi)
            env.c2, env.c1, env.t1b, env.t1a = _ctmrg_square_2x2_step(env.c2, env.c1, env.t2b, env.t2a, env.t4b, env.t4a, env.t1b, env.t1a, a, b, D, chi)
            env.c3, env.c2, env.t2a, env.t2b = _ctmrg_square_2x2_step(env.c3, env.c2, env.t3a, env.t3b, env.t1a, env.t1b, env.t2a, env.t2b, a, b, D, chi)
            env.c4, env.c3, env.t3b, env.t3a = _ctmrg_square_2x2_step(env.c4, env.c3, env.t4b, env.t4a, env.t2b, env.t2a, env.t3b, env.t3a, a, b, D, chi)
            
        if env2 is not None:
            curErr = convergenceTestFct(env, env2)
            if curErr < err:
                return env, curErr
    
        env2 = copy(env)

    return env, curErr


class CTMEnv1x1InvSymm:
    """
    c----t1----c*
    |     |    |
    t2--    --t2*
    |          |
    c*---t1*---c
    """
    def __init__(self, D0, D1, chi, dtype):
        self.c = np.ones((chi, chi), dtype=dtype)
        self.t1 = np.ones((chi, D0, chi), dtype=dtype)
        self.t2 = np.ones((chi, D1, chi), dtype=dtype)
    def toarray1x1(self):
        t = tdot(
            tdot(self.c, self.t1, [1,0]),
            tdot(self.c, self.t2, [1,0]).conj(),
            [2,0])
        return tdot(t, t.conj(), [[3,0],[0,3]])
    def toarray1x2(self):
        t = tdot(self.c, self.t1, [1,0])
        t = tdot(t, t.conj(), [2,2])
        t = tdot(t, self.t2.conj(), [2,0])
        return tdot(t, t.conj(), [[0,4],[4,0]])
    def halfplane_upper(self, num_links):
        x = self.c
        for j in xrange(num_links):
            x = tdot(x, self.t1, [1+j,0])
        return tdot(x, self.c.conj(), [num_links+1,0])
    def halfplane_lower(self, num_links):
        x = self.c
        for j in xrange(num_links):
            x = tdot(x, self.t1.conj(), [1+j,0])
        return tdot(x, self.c.conj(), [num_links+1,0])
    def halfplane_left(self, num_links):
        x = self.c.conj()
        for j in xrange(num_links):
            x = tdot(x, self.t2, [1+j,0])
        return tdot(x, self.c, [num_links+1,0])
    def halfplane_right(self, num_links):
        x = self.c.conj()
        for j in xrange(num_links):
            x = tdot(x, self.t2.conj(), [1+j,0])
        return tdot(x, self.c, [num_links+1,0])
    def lift(self, chi2):
        u = np.fromfunction(np.vectorize(lambda j,k: 1.0 if j==k else 0.0), (chi2, self.c.shape[0]))
        r = CTMEnv1x1InvSymm(self.t1.shape[1], self.t2.shape[1], chi2, type(self.c[0,0]))
        r.c = dot(dot(u, self.c), u.transpose())
        r.t1 = tdot(tdot(u, self.t1, [1,0]), u, [2,1])
        r.t2 = tdot(tdot(u, self.t2, [1,0]), u, [2,1])
        return r

def ctmrg_square_1x1_invsymm(a, chi, err=1e-6, env=None, max_iterations=10000000, iteration_bunch=1000, verbose=False):
    """
    CTMRG implementation on a square lattice for a one site unit cell, which has a "hermitian"
    inversion symmetry, i.e.
    
       i           k            i
       |           |            |
    l--a--j  =  l--a*--j  =  j--a*--l.
       |           |            |
       k           i            k
    The idea of this implementation is described in PRB 85, 205117 (2012) by Roman Orus.
    
    One iteration of this procedure has a runtime complexity of O(chi**3 D**3 + chi**2 D**4) and
    a memory complexity of O(chi**2 D**3).
    """
    D0 = a.shape[0]
    D1 = a.shape[1]
    if not isinstance(env, CTMEnv1x1InvSymm):
        env = CTMEnv1x1InvSymm(D0, D1, chi, type(a[0,0,0,0]))
    
    c = env.c
    t1 = env.t1
    t2 = env.t2
    s3 = s4 = np.zeros(chi)
    env2 = None
    
    for j in xrange(max_iterations/iteration_bunch):
        for j2 in xrange(iteration_bunch):
            # horizontal move
            u,s1,c = svd(tdot(c, t1, [1,0]).reshape(D0*chi, chi), full_matrices=False)
            s1 /= dot(s1,s1) # fun fact: converges slightly faster if singular values are normalised
            c = dot(np.diag(s1), c)
            t2 = tdot(t2, a, [1,3]).swapaxes(1,4).swapaxes(2,3).swapaxes(3,4)
            t2 = t2.reshape(D0*chi, D1, D0*chi)
            t2 = tdot(tdot(u.conj(), t2, [0,0]), u, [2,0])
            
            # vertical move
            c,s2,u = svd(tdot(t2, c, [2,0]).reshape(chi, D1*chi), full_matrices=False)
            s2 /= dot(s2,s2) # fun fact: converges slightly faster if singular values are normalised
            c = dot(c, np.diag(s2))
            t1 = tdot(a, t1, [0,1]).swapaxes(0,2).swapaxes(1,3).swapaxes(2,3).reshape(D1*chi, D0, D1*chi)
            t1 = tdot(tdot(u, t1, [1,0]), u.conj(), [2,1])
            
            # numerical renormalisation (just for stability reasons)
            c /= np.max(np.abs(c))
            t1 /= np.max(np.abs(t1))
            t2 /= np.max(np.abs(t2))
        
        curErr = np.max([np.abs(s1-s3), np.abs(s2-s4)])
        if verbose:
            print "[ctmrg_square_1x1_invsymm] error after {:d} iterations is {:e}".format((j+1)*iteration_bunch, curErr)
            
        if curErr < err:
            env.c = c
            env.t1 = t1
            env.t2 = t2
            return env, env2, curErr, (j+1)*iteration_bunch
        
        s3 = s1
        s4 = s2
        env2 = copy(env)
    
    env.c = c
    env.t1 = t1
    env.t2 = t2
    return env, env2, curErr, max_iterations



class CTMEnvironment1x1:
    """
    C1---T1---C2
    |     |    |
    T4--    --T2
    |          |
    C4---T3---C3
    
    C1--1   0--C2       0   1
    |           |       |   |
    0           1   1--C3   C4--0
    
                    0                 2
    0--T1--2        |        1        |
       |        1--T2        |        T4--1
       1            |     2--T3--0    |
                    2                 0
    """
    def __init__(self, D0, D1, chi, dtype):
        self.c1 = np.ones((chi, chi), dtype=dtype)
        self.c2 = np.ones((chi, chi), dtype=dtype)
        self.c3 = np.ones((chi, chi), dtype=dtype)
        self.c4 = np.ones((chi, chi), dtype=dtype)
        self.t1 = np.ones((chi, D0, chi), dtype=dtype)
        self.t2 = np.ones((chi, D1, chi), dtype=dtype)
        self.t3 = np.ones((chi, D0, chi), dtype=dtype)
        self.t4 = np.ones((chi, D1, chi), dtype=dtype)
    def toarray1x1(self):
        return tdot(
            tdot(
                tdot(self.c1, self.t1, [1,0]),
                tdot(self.c2, self.t2, [1,0]),
                [2,0]),
            tdot(
                tdot(self.c3, self.t3, [1,0]),
                tdot(self.c4, self.t4, [1,0]),
                [2,0]),
            [[3,0],[0,3]])
    def toarray1x2(self):
        return tdot(
            tdot(
                tdot(tdot(self.c1, self.t1, [1,0]), self.t1, [2,0]),
                tdot(self.c2, self.t2, [1,0]),
                [3,0]),
            tdot(
                tdot(tdot(self.c3, self.t3, [1,0]), self.t3, [2,0]),
                tdot(self.c4, self.t4, [1,0]),
                [3,0]),
            [[4,0],[0,4]])
    def halfplane_upper(self, num_links):
        x = self.c1
        for j in xrange(num_links):
            x = tdot(x, self.t1, [1+j,0])
        return tdot(x, self.c2, [num_links+1,0])
    def halfplane_lower(self, num_links):
        x = self.c3
        for j in xrange(num_links):
            x = tdot(x, self.t3, [1+j,0])
        return tdot(x, self.c4, [num_links+1,0])
    def halfplane_left(self, num_links):
        x = self.c4
        for j in xrange(num_links):
            x = tdot(x, self.t4, [1+j,0])
        return tdot(x, self.c1, [num_links+1,0])
    def halfplane_right(self, num_links):
        x = self.c2
        for j in xrange(num_links):
            x = tdot(x, self.t2, [1+j,0])
        return tdot(x, self.c3, [num_links+1,0])
            
    def to_2x2(self, e2x2):
        e2x2.c1 = self.c1
        e2x2.c2 = self.c2
        e2x2.c3 = self.c3
        e2x2.c4 = self.c4
        e2x2.t1b = e2x2.t1a = self.t1
        e2x2.t2b = e2x2.t2a = self.t2
        e2x2.t3b = e2x2.t3a = self.t3
        e2x2.t4b = e2x2.t4a = self.t4

def _ctmrg_square_1x1_step_directional_single(c1, c4, t1, t3, t4, a, D0, D1, chi):
    c1 = tdot(c1, t1, [1,0]).reshape(chi*D0, chi)
    c4 = tdot(t3, c4, [2,0]).swapaxes(1,2).reshape(chi, chi*D0)
    t4 = tdot(t4, a, [1,3]).swapaxes(1,4).swapaxes(3,4).swapaxes(2,4).reshape(chi*D0, D1, chi*D0)
    
    w,z = np.linalg.eigh(dot(c1, c1.conj().transpose()) + dot(c4.conj().transpose(), c4))
    idx = w.argsort()[::-1]
    w = w[idx][:chi]
    w /= np.dot(w,w)
    z = z[:,idx][:,:chi]
    c1 = dot(z.conj().transpose(), c1)
    c4 = dot(c4, z)
    t4 = tdot(tdot(z.conj(), t4, [0,0]), z, [2,0])
    
    c1 /= np.max(np.abs(c1))
    c4 /= np.max(np.abs(c4))
    t4 /= np.max(np.abs(t4))
    return c1, c4, t4, w
def _ctmrg_square_1x1_step_directional(c1, c2, c3, c4, t1, t2, t3, t4, a, a2, a3, a4, D0, D1, chi):
    # note: performing moves in this order preserves c1=c3, c2=c4 and c1=conj(c2) if a is inversion symmetric
    c1, c4, t4, s1 = _ctmrg_square_1x1_step_directional_single(c1, c4, t1, t3, t4, a,  D0, D1, chi)
    c3, c2, t2, s2 = _ctmrg_square_1x1_step_directional_single(c3, c2, t3, t1, t2, a3, D0, D1, chi)
    c2, c1, t1, s3 = _ctmrg_square_1x1_step_directional_single(c2, c1, t2, t4, t1, a2, D1, D0, chi)
    c4, c3, t3, s4 = _ctmrg_square_1x1_step_directional_single(c4, c3, t4, t2, t3, a4, D1, D0, chi)
    return c1,c2,c3,c4,t1,t2,t3,t4,np.concatenate([s1,s2,s3,s4])

def _ctmrg_square_1x1_step_radial(c1, c2, c3, c4, t1, t2, t3, t4, a, a2, a3, a4, D0, D1, chi):
    c1 = tdot(t4, c1, [2,0]).swapaxes(1,2).reshape(chi, D1*chi)
    c2 = tdot(t1, c2, [2,0]).swapaxes(1,2).reshape(chi, D0*chi)
    c3 = tdot(t2, c3, [2,0]).swapaxes(1,2).reshape(chi, D1*chi)
    c4 = tdot(t3, c4, [2,0]).swapaxes(1,2).reshape(chi, D0*chi)
    t1 = tdot(t1, a, [1,0]).swapaxes(1,4).swapaxes(2,3).swapaxes(3,4).reshape(D1*chi, D0, D1*chi)
    t2 = tdot(t2, a, [1,1]).swapaxes(1,2).swapaxes(2,4).swapaxes(3,4).reshape(D0*chi, D1, D0*chi)
    t3 = tdot(t3, a, [1,2]).swapaxes(1,3).reshape(D1*chi, D0, D1*chi)
    t4 = tdot(t4, a, [1,3]).swapaxes(1,4).swapaxes(2,3).reshape(D0*chi, D1, D0*chi)
    c1 = tdot(c1, t1, [1,0]).reshape(D0*chi, D1*chi)
    c2 = tdot(c2, t2, [1,0]).reshape(D1*chi, D0*chi)
    c3 = tdot(c3, t3, [1,0]).reshape(D0*chi, D1*chi)
    c4 = tdot(c4, t4, [1,0]).reshape(D1*chi, D0*chi)
    x1,x2 = dot(c1, c2), dot(c3, c4)
    y1,y2 = dot(c4, c1), dot(c2, c3)
    u12,s12,v12 = svd(dot(y2,y1))
    u23,s23,v23 = svd(dot(x2,x1))
    u34,s34,v34 = svd(dot(y1,y2))
    u41,s41,v41 = svd(dot(x1,x2))
    s12,s23,s34,s41 = s12[:chi],s23[:chi],s34[:chi],s41[:chi]
    u12,u23,u34,u41 = u12[:,:chi],u23[:,:chi],u34[:,:chi],u41[:,:chi]
    v12,v23,v34,v41 = v12[:chi],v23[:chi],v34[:chi],v41[:chi]
    s12 /= dot(s12,s12)
    s23 /= dot(s23,s23)
    s34 /= dot(s34,s34)
    s41 /= dot(s41,s41)
    c1 = dot(dot(u41.transpose().conj(), c1), v12.transpose().conj())
    c2 = dot(dot(u12.transpose().conj(), c2), v23.transpose().conj())
    c3 = dot(dot(u23.transpose().conj(), c3), v34.transpose().conj())
    c4 = dot(dot(u34.transpose().conj(), c4), v41.transpose().conj())
    t1 = tdot(tdot(v12, t1, [1,0]), u12, [2,0])
    t2 = tdot(tdot(v23, t2, [1,0]), u23, [2,0])
    t3 = tdot(tdot(v34, t3, [1,0]), u34, [2,0])
    t4 = tdot(tdot(v41, t4, [1,0]), u41, [2,0])
    c1 /= np.max(np.abs(c1))
    c2 /= np.max(np.abs(c2))
    c3 /= np.max(np.abs(c3))
    c4 /= np.max(np.abs(c4))
    t1 /= np.max(np.abs(t1))
    t2 /= np.max(np.abs(t2))
    t3 /= np.max(np.abs(t3))
    t4 /= np.max(np.abs(t4))
    return c1,c2,c3,c4,t1,t2,t3,t4,np.concatenate([s12,s23,s34,s41])

def ctmrg_square_1x1(a, chi, err=1e-6, max_iterations=1000000, min_iterations=100, env=None, step_method="directional"):
    D0 = a.shape[0]
    D1 = a.shape[1]
    if not isinstance(env, CTMEnvironment1x1):
        env = CTMEnvironment1x1(D0, D1, chi, type(a[0,0,0,0]))
    
    a2 = a.swapaxes(0,1).swapaxes(1,2).swapaxes(2,3)
    a3 = a.swapaxes(0,2).swapaxes(1,3)
    a4 = a2.swapaxes(0,2).swapaxes(1,3)
    
    if step_method == "directional":
        step = _ctmrg_square_1x1_step_directional
    elif step_method == "radial":
        step = _ctmrg_square_1x1_step_radial
    
    env2 = None
    curErr = None
    s2 = np.zeros(4*chi)
    
    #for j in xrange(min_iterations):
    #    env.c1,env.c2,env.c3,env.c4,env.t1,env.t2,env.t3,env.t4,s2 = step(
    #        env.c1, env.c2, env.c3, env.c4, env.t1, env.t2, env.t3, env.t4, a, a2, a3, a4, D0, D1, chi)
    #env2 = copy(env)
    
    for j in xrange(max_iterations/min_iterations):
        for j2 in xrange(min_iterations):
            env.c1,env.c2,env.c3,env.c4,env.t1,env.t2,env.t3,env.t4,s = step(
                env.c1, env.c2, env.c3, env.c4, env.t1, env.t2, env.t3, env.t4, a, a2, a3, a4, D0, D1, chi)
        
        curErr = np.max(np.abs(s-s2))
        if curErr < err:
            return env, env2, curErr, (j+1)*min_iterations
        env2 = copy(env)
        s2 = s
    
    return env, env2, curErr, max_iterations


