#
# Quantum Particle Simulation for One Sided Step Potential
#   by Aidan Macdonald
#
# Using the notation and analysis from
# "Principles of Quantum Mechanics" by R. Shankar
#   See pg. 167
#


from numpy import *
from matplotlib.pyplot import *
from matplotlib import animation
from multiprocessing import Pool
from itertools import repeat, izip
from time import sleep

h_bar = 1.0
m     = 1.0
V0    = 3.0
delta = 1.0
a     = -10.0
k0    = -2.0


A = 1.0/sqrt(2*pi)
i = 1j

def theta(x):
    return 1.0*(x > 0)

# Eigen-functions
def psi(k1, x):
    return A*( (exp(i*k1*x) + BA(k1)*exp(-i*k1*x))*theta(-x) + 
               CA(k1)*exp(i*sqrt(real(k1**2 - 2*m*V0/h_bar) + 0j)*x)*theta(x) )

# Eigen-values
def E(k1):
    return k1**2*h_bar**2/(2*m)


# Ratios
def BA(k1):
    E1 = E(k1)
    return (sqrt(E1) - sqrt(E1 - V0))/(sqrt(E1) + sqrt(E1 - V0))

def CA(k1):
    E1 = E(k1)
    return 2*sqrt(E1)/(sqrt(E1) + sqrt(E1 - V0))

# Initial projection
def proj(k1):
    return (delta**2/pi)**0.25 * exp(-(k1 - k0)**2 * delta**2/2) * exp(i*k1*a)


# Propagated Wave-function
def prop(x, t, k1):
    return proj(k1)*exp(-i*E(k1)*t/h_bar)*psi(k1, x)


# Wave as a function of space and time
# Notice that a(k1) falls off fast far from k0, Var(a) = 1/delta**2
#        and  E(k1) falls off fast far from 0,  Var(E in prop) = m/(t*h_bar)
# Thus the important integration region becomes
#   [-3/delta, 3*sqrt(m/(t*h_bar))] ie. 3 STDEV's from either
def wave(x, t): # t is a scalar
    k1 = arange(-5/delta, 5/delta, 0.01, dtype=complex)
    
    k1 = k1.reshape((k1.size, 1))
    x  =  x.reshape((1,       x.size))
    
    return sum(prop(x, t, k1)*0.01, axis=0).flatten()


#
##
## Animation Section  
##
#

def per_time((x, t)):
    return absolute(wave(x, t))

mp_pool = Pool(4)

x = arange(-20, 20, 0.01, dtype=complex)
t = arange(0, 20, 0.1)

try:
    f = memmap("one-side-step.dat", mode="r", 
               shape=(t.size, x.size), dtype=float)
except:
    f = memmap("one-side-step.dat", mode="w+",
               shape=(t.size, x.size), dtype=float)
    for n, p in enumerate(mp_pool.imap(per_time, 
                                       izip(repeat(x), t))):
        f[n, :] = p

def data_gen():
    for n, tn in enumerate(t):
        yield real(x), f[n]
        sleep(.1)

fig, ax = subplots()
line,   = ax.plot([], [], lw=2)
ax.set_ylim(-0.1, 5.1)
ax.set_xlim(-20, 20)
ax.grid()

def run((x, p)):
    line.set_data(x, p)
    return line,


ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=10,
                              repeat=True)
show()
