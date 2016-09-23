#
# Simulation of a Gaussian Particle in One Sided Step Potential
#   using the Path Integral Formulation
#
#  by Aidan Macdonald
#
#


from numpy import *
from numpy.linalg import matrix_power
from matplotlib.pyplot import *
from matplotlib import animation
from multiprocessing import Pool
from itertools import repeat, izip
from time import sleep
i = 1j


# Parameters
h_bar = 1.0
m     = 1.0
V0    = 0.0
delta = 1.0
a     = 10.0
k0    = -2.0

# Space
dx = 0.01
x = arange(-20, 20, dx)
x = x.reshape((x.size, 1))

# Time
dt = 0.01
t = arange(0, 20, dt)

# Initial Wave Function
psi = (pi*delta**2)**(-0.25) * exp(i*k0*(x + a))*exp(-(x+a)**2/(2*delta**2))

# Transition Matrix
dE = dx*exp(i/h_bar * (m*(x - x.T)**2/(2*dt) + V0*((x + x.T)/2 > 0)*dt ))
E = matrix_power(dE, 4)

def data_gen():
    psi_cur = psi
    for n, tn in enumerate(t):
        p = absolute(psi_cur)**2
        yield real(x), p/(sum(p)*dx)
        psi_cur = E.dot(psi_cur)


fig, ax = subplots()
line,   = ax.plot([], [], lw=2)
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(-20, 20)
ax.grid()

def run((x, p)):
    line.set_data(x, p)
    return line,


ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=100, repeat=True)
show()





