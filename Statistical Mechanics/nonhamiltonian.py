"""
Simulation as demonstration of the work,
  Tuckerman, M. E., C. J. Mundy, and G. J. Martyna. "On the 
  classical statistical mechanics of non-Hamiltonian systems." 
  EPL (Europhysics Letters) 45.2 (1999): 149.

I am simulating the problem mentioned at the end of the paper
called the "Gaussian isokinetic system"

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

N, time_steps = int(1e5), int(1e5)
#N, time_steps = int(1e6), int(1e6)
q, p = np.random.randn(N), np.random.randn(N)

# Potential and 
phi = lambda q: 0
F   = lambda q: 0
alpha = np.sum(p * F(q))/np.sum(p * p)

# Dynamical equations
dq = lambda q, p: p
dp = lambda q, p: F(q) - alpha * p
dt = 1e-5

# Metric
g = lambda q, p: np.exp(-phi(q)*(N-1)/np.sum(p*p))

S = []
for i in range(time_steps):
    p += dp(q, p) * (dt/2)
    q += dq(q, p) * dt
    p += dp(q, p) * (dt/2)
    
S = np.array(S)

plt.plot(S)
plt.show()
