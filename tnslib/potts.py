import numpy as np

#def _delta(j,k):
#    return 1.0 if j==k else 0.0

def interaction_term(q, J):
    """
    gives the matrix elements h_{jklm} of the quantum potts interaction term
    <j,k| (J sum_{mu=1}^{q} |mu,mu><mu,mu|) |l,m>
    """
    return np.fromfunction(np.vectorize(lambda j,k,l,m: J if j==k==l==m else 0.0), (q,q,q,q), dtype=int)

def externalfield_term(q, h):
    """
    <j| h/q sum_{mu=1}^{q} sum_{mu,nu} (1 - delta_{mu,nu}) |mu><nu| |k>
    """
    #def _delta(j,k):
    #    return 1.0 if j==k else 0
    #return h*np.fromfunction(np.vectorize(lambda j,k: (1.0 - _delta(j,k))/q), (q,q), dtype=int)
    return np.fromfunction(np.vectorize(lambda j,k: 0.0 if j==k else 1.0*h/q), (q,q), dtype=int)

def magnetisationz_operator(q, j):
    """
    gives the one-particle operator |j><j| for measuring the z-magnetisation
    """
    return np.fromfunction(np.vectorize(lambda k,l: 1.0 if j==k==l else 0), (q,q), dtype=int)

