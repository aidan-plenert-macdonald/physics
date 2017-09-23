"""
Simulations of various Ensembles

See,
Andersen, Hans C. "Molecular dynamics simulations at constant pressure and/or temperature." The Journal of chemical physics 72.4 (1980): 2384-2393.

"""

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

class nPTEnsemble:
    def __init__(self, H):
        self.H = H

    def generate(self, n, P, T):
        q, V = np.random.rand(n, 3), n**(1.0/3)
        while True:
            p = np.random.randn(n, 3)*np.sqrt(T/2.0)
            _q, _p = self.H.leapfrog(q, p, steps=np.random.randint(2, 5))
            if np.random.rand() < np.exp((self.H(q, p) - self.H(_q, _p))/T):
                q, p = _q, _p
            _V = V + 0.5
            yield q, p, collisions

    def V(self, n, P, T):
        return V

    def S(self, n, P, T):
        pass

    def mu(self, n, P, T):
        pass
        

nVTIdealGas = nVTEnsemble(FreeParticle)

print [nVTIdealGas.P(10000, 10.0**i, 1.0) for i in range(10)]
