import numpy as np
from scipy.linalg import expm

delta = np.array([[1.,0],[0,1]])
delta.flags.writeable = False

sigmax = np.array([[0.0, 1.0], [1.0, 0.0]])
sigmax.flags.writeable = False

sigmaz = np.array([[-1.0, 0.0], [0.0, 1.0]])
sigmaz.flags.writeable = False

"""
<j,k| sigmaz sigmaz |l,m>
"""
sigmazsigmaz = np.fromfunction(lambda j,k,l,m: delta[j,l]*delta[k,m]*(-1)**(j+k), (2,2,2,2), dtype=int)
sigmazsigmaz.flags.writeable = False

def exp_sigmax(alpha):
    """
    gives the tensor-elements g_{j1,j2} of the one-particle gate
    <j1|exp(alpha sigma_x)|j2>
    """
    c = np.cosh(alpha)
    s = np.sinh(alpha)
    return np.array([[c,s],[s,c]])

def exp_sigmaz(alpha):
    """
    gives the tensor-elements g_{j1,j2} of the one-particle gate
    <j1|exp(alpha sigma_z)|j2>
    """
    return np.array([[np.exp(-alpha),0],[0,np.exp(alpha)]])

def exp_sigmax_sigmax(alpha):
    """
    gives the tensor-elements g_{j1,j2,k1,k2} of the two-particle gate
    <j1,k1|exp(alpha sigma_x sigma_x)|j2,k2>
    """
    g = np.zeros((2,2,2,2), dtype=type(alpha))
    g[0,0,0,0] = g[0,1,0,1] = g[1,0,1,0] = g[1,1,1,1] = np.cosh(alpha)
    g[0,0,1,1] = g[1,1,0,0] = g[0,1,1,0] = g[1,0,0,1] = np.sinh(alpha)
    return g

def exp_sigmaz_sigmaz(alpha):
    """
    gives the tensor-elements g_{j,k,l,m} of the two-particle gate
    <j,l|exp(alpha sigma_z sigma_z)|k,m>

     j   k
     |   |
    +-----+
    |  g  |
    +-----+
     |   |
     l   m
    """
    g = np.zeros((2,2,2,2), dtype=type(alpha))
    g[0,0,0,0] = g[1,1,1,1] = np.exp(alpha)
    g[0,1,0,1] = g[1,0,1,0] = np.exp(-alpha)
    return g

def exp_sigmaz_sigmaz_mpo(alpha):
    """
        i
        |
       +-+
    l--|g|--j
       +-+
        |
        k
    """
    if alpha < 0:
        alpha = alpha*1j
    c = np.sqrt(np.cosh(alpha))
    s = np.sqrt(np.sinh(alpha))
    w = np.array([[c,s],[c,-s]])
    w2 = np.fromfunction(lambda j,k,l: delta[j,k]*w[j,l], (2,2,2), dtype=int)
    return np.einsum(w2, [0,4,1], w2.conj(), [4,2,3])
    
    #f = np.array([np.sqrt(np.sinh(alpha)),np.sqrt(np.cosh(alpha))])
    #if alpha > 0:
    #    return np.fromfunction(lambda i,j,k,l: delta[i,k]*f[j]*f[l]*(-1)**((j+l)*i), (2,2,2,2), dtype=int)
    #else:
    #    raise ValueError("can not handle negative parameter! would give complex output!")

def exp_sigmaz_sigmaz_pepo_square(alpha):
    m = exp_sigmaz_sigmaz_mpo(alpha)
    return np.einsum(m, [0,2,6,3], m, [6,4,1,5])


def imtime_evolution_from_pair_hamiltonian_mpo(h1, h2, tau):
    p = h1.shape[0]
    u1 = expm(-0.5*tau * h1)
    u2 = expm(-tau * h2.reshape(p*p, p*p)).reshape(p,p,p,p).swapaxes(1,2).reshape(p*p, p*p)
    r = np.linalg.matrix_rank(u2)
    u,s,v = np.linalg.svd(u2)
    u = np.dot(u[:,:r], np.diag(np.sqrt(s[:r]))).reshape(p,p,r)
    v = np.dot(np.diag(np.sqrt(s[:r])), v[:r]).reshape(r,p,p)
    g = np.einsum(u, [0,4,1], v, [3,4,2])
    g = np.einsum(u1, [0,4], g, [4,1,5,3], u1, [5,2])
    return g

def imtime_evolution_from_pair_hamiltonian_pepo(h1, h2, tau):
    g = imtime_evolution_from_pair_hamiltonian_mpo(h1, h2, tau)
    return np.einsum(g, [0,2,6,3], g, [6,4,1,5])

