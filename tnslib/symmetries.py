import numpy as np
from fractions import Fraction

def translationSymmetry(n, spinsPerSite=1):
    def translate(x):
        #return ((x<<spinsPerSite)&((1<<(spinsPerSite*n))-1)) | 
        return (x>>spinsPerSite) | ((x&1)<<(spinsPerSite*(n-1)))
    return translate

def spinFlipSymmetry(n, spinsPerSite=1):
    def spinFlip(x):
        return (~x) & ((1<<(spinsPerSite*n))-1)
    return spinFlip

def _generate(x, symmetries, res):
    S = symmetries[0]
    if len(symmetries) == 1:
        if x in res:
            return
        else:
            res.append(x)
    else:
        _generate(x, symmetries[1:], res)
    y = S(x)
    while y != x:
        if len(symmetries) == 1:
            res.append(y)
        else:
            _generate(y, symmetries[1:], res)
        y = S(y)

def _seqLens(x, symmetries):
    L = [1] * len(symmetries)
    for j in range(len(symmetries)):
        S = symmetries[j]
        y = S(x)
        while x != y:
            y = S(y)
            L[j] += 1
    return L

def _applySymmetries(x, symmetries, n):
    for j in range(len(n)):
        S = symmetries[j]
        for k in range(n[j]):
            x = S(x)
    return x

class Symmetries:
    def __init__(self, symmetryOps, n, spinsPerSite=1):
        self.n = n
        self.spinsPerSite = spinsPerSite
        self.ops = map(lambda f: f(n, spinsPerSite), symmetryOps)
        self._representatives = None
        self._blochVectors = None
    def __getitem__(self, i):
        return self.ops[i]
    def __len__(self):
        return len(self.ops)
    def generateClass(self, x):
        res = []
        _generate(x, self, res)
        return res
    def isRepresentative(self, x):
        return x == min(self.generateClass(x))
    def representatives(self):
        if self._representatives is None:
            self._representatives = filter(self.isRepresentative, xrange(2**(self.spinsPerSite*self.n)))
        return self._representatives
    def blochVectorsForRepresentative(self, r):
        res = dict()
        L = _seqLens(r, self.ops)
        Larray = np.array(L, dtype=float)
        for j in np.ndindex(tuple(L)):
            v = np.zeros(2**self.n, dtype=complex)
            for m in np.ndindex(tuple(L)):
                k = _applySymmetries(r, self.ops, m)
                v[k] += np.exp(-2.0j * np.pi * np.dot(np.array(j) / Larray, np.array(m)))
            if not (np.abs(v) < 1e-10).all():
                key = tuple(Fraction(j[l], L[l]) for l in xrange(len(L)))
                if key in res:
                    res[key].append(v / np.sqrt(np.vdot(v, v)))
                else:
                    res[key] = [ v / np.sqrt(np.vdot(v, v)) ]
        return res
    def blochVectors(self):
        if self._blochVectors is None:
            res = dict()
            for r in self.representatives():
                tmp = self.blochVectorsForRepresentative(r)
                for key in tmp:
                    if key in res:
                        res[key] += tmp[key]
                    else:
                        res[key] = tmp[key]
            resKeysSorted = sorted(res)
            self._blochVectorsInfo = map(lambda x: (len(res[x]), np.array(x, dtype=float) * 2.0 * np.pi), resKeysSorted)
            self._blochVectors = np.ndarray((2**self.n, 2**self.n), dtype=complex)
            i = 0
            for k in resKeysSorted:
                for v in res[k]:
                    self._blochVectors[:,i] = v
                    i += 1
        return self._blochVectorsInfo, self._blochVectors
    def blockDiagonalise(self, A, mode="blocks_only"):
        Uinfo, U = self.blochVectors()
        B = np.dot(np.dot(np.transpose(np.conj(U)), A), U)
        if mode == "full_matrix":
            return B
        else: # mode == "blocks_only"
            blockLens = map(lambda x: x[0], self._blochVectorsInfo)
            blockInfo = map(lambda x: x[1], self._blochVectorsInfo)
            blocks = []
            i = 0
            for j in blockLens:
                blocks.append(B[i:i+j,i:i+j])
                i += j
            return blockInfo, blocks
    def getSymmetryOperationAsMatrix(self, i):
        m = np.zeros((2**self.n, 2**self.n), dtype=float)
        S = self.ops[i]
        for j in range(2**self.n):
            m[S(j),j] = 1
        return m

# e.g.:
# symms = Symmetries([translationSymmetry, spinFlipSymmetry], 4)

