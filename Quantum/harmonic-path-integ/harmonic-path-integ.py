#
# Gaussian Particle in Harmonic Oscillator
#  See http://physics.ucsd.edu/students/courses/spring2015/physics142/Lectures/Lecture2/Lecture2.pdf
#
#

from numpy import *
from numpy.linalg import matrix_power
from matplotlib.pyplot import *
from matplotlib import animation

dx = 8.0/600
x = arange(-4, 4, dx)
x = x.reshape((x.size, 1))
x0 = 0.75

T0 = 2*pi
dt = T0/128

alpha = 2.0
i = 1.0j

psi = (2.0/pi)**(0.25) * exp(i*x) * exp(-(alpha/2)*(x - x0)**2)

def V(x):
    return x**2


K = exp(i * ((x - x.T)**2/dt - V((x + x.T)/2)*dt )/2)*dx
Kprop = matrix_power(K, 1)

def data_gen():
    psi_cur = psi
    yield real(x), absolute(V(x))
    for n in xrange(256):
        p = absolute(psi_cur)**2
        P = sum(p)*dx
        psi_cur = Kprop.dot(psi_cur/sqrt(P))
        yield real(x), p/P


fig, ax = subplots()
line,   = ax.plot([], [], lw=2)
ax.set_ylim(-0.1, 1.5)
ax.set_xlim(-3, 3)
ax.grid()
title('Path Integral of a Gaussian Particle in Harmonic Oscillator')

def run((x, p)):
    line.set_data(x, p)
    return line,


ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=100, repeat=True)
show()
#ani.save('harmonic-path-integ.mp4', writer='ffmpeg')
                                            
