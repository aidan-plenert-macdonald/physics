#
# Ideal Gas Simulation
#  by Aidan Macdonald
#
# I simulate the evolution of an ideal
# gas in a fixed volume and compute various
# Microcanonical values.
#
# Pressure can be computed from particle
# collisions. Entropy is given by Shannon
# and Volume and Energy are controlled.
# 
# 

from numpy import *
from scipy.stats import entropy
import h5py

N = 10000
R = 3.0

steps = 10000
dt = 0.01

# Random momemtums
p = 10*random.randn(N, 3)
q = zeros((N, 3))


P = []
V = []
S = []
E = []
for i in range(steps):
    q += dt*p
    
    # Boundary collision handling
    L = linalg.norm(q, axis=1)
    dp = -q*(sum(p*q, axis=1)/linalg.norm(q, axis=1)**2*(L >= R)).reshape(-1, 1)
    p += dp
    q *= (R*(L >= R)/L + (L < R)).reshape(-1, 1)
    
    # Compute Thermodynamics
    P.append(linalg.norm(2*dp)/(4*pi*R**2))
    V.append(4*pi*R**3/3)
    S.append(entropy(histogramdd(q, int(sqrt(N)))[0].flatten()) + entropy(histogramdd(p, int(sqrt(N)))[0].flatten()))
    E.append(sum(0.5*p**2))
    
    R *= 1.001
    if i % (steps/100) == 0:
        print "~~ %f%s complete ~~" % (100*float(i)/steps, '%')

# T = dE/dS
T = nan_to_num(diff(E)/diff(S))

# p = T dS/dV
print polyfit(P[1:], T*diff(S)/diff(V), 3)

f = h5py.File('ideal_gas.h5')
f.create_dataset('P', data=P)
f.create_dataset('V', data=V)
f.create_dataset('S', data=S)
f.create_dataset('E', data=E)

    




