import numpy as np
from itertools import islice

class Hamiltonian:
    def __init__(self, E, dq, dp):
        self._E, self.dq, self.dp = E, dq, dp

    def leapfrog(self, q, p, steps=1, dt=1e-3):
        for i in range(steps):
            p += self.dp(q, p) * (dt/2)
            q += self.dq(q, p) * dt
            p += self.dp(q, p) * (dt/2)
        return q, p

    def __call__(self, q, p):
        return self._E(q, p)

FreeParticle = Hamiltonian(lambda q, p: 0.5*np.sum(p**2),
                           lambda q, p: p,
                           lambda q, p: 0.0)

class nVTEnsemble:
    def __init__(self, H):
        self.H = H

    def generate(self, n, V, T):
        bound = V**(1.0/3)
        q = np.random.rand(n, 3)*bound
        collisions = 0
        while True:
            p = np.random.randn(n, 3)*np.sqrt(T/2.0)
            _q, _p = self.H.leapfrog(q, p, steps=np.random.randint(2, 5))
            lout, hout = _q < 0, _q > 0
            _q[lout], _q[hout] = -_q[lout], bound - _q[hout]
            _p[lout], _p[hout] = -_p[lout], -_p[hout]
            if np.random.rand() < np.exp((self.H(q, p) - self.H(_q, _p))/T):
                q, p, collisions = _q, _p, np.sum(1.0*np.logical_or(np.any(lout, axis=1), np.any(hout, axis=1)))
            yield q, p, collisions

    def n(self, n, V, T):
        return n
    
    def V(self, n, V, T):
        return V

    def T(self, n, V, T):
        return T

    def P(self, n, V, T):
        f, Z = 0, 0
        gen = self.generate(n, V, T)
        while True:
            df, dZ = 0, 0
            for q, p, coll in islice(gen, 100):
                df += coll
                dZ += 1
            
            if Z != 0 and ((f + df)/(Z + dZ) - f/Z)/(f/Z) < 1e-2:
                return T * (f + df)/(Z + dZ)
            f, Z = f+df, Z+dZ
            
    def S(self, n, V, T):
        pass

    def mu(self, n, V, T):
        pass
        

nVTIdealGas = nVTEnsemble(FreeParticle)

print [nVTIdealGas.P(10000, 10.0**i, 1.0) for i in range(10)]
