#
# Transistor Simulation
#  by Aidan Macdonald
#
#

from numpy import *
from numpy.ctypeslib import ndpointer
from time import sleep
import ctypes, h5py

f = h5py.File('transistor.h5', 'w')

lib = ctypes.cdll.LoadLibrary('./transistor.so')
evolve = lib.evolve
evolve.restype = None
evolve.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                   ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                   ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                   ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                   ctypes.c_double, ctypes.c_double, ctypes.c_int]

dx = 0.05
dt = dx**2
V0 = 10.0

width  = 1.0
height = 0.2
p_mom = 5.0

x = arange(-10, 10, dx).reshape(-1, 1)
y = arange(-10, 10, dx).reshape(1, -1)

psi = exp(-(x**2 + (y+4)**2)/2)*exp(1j*p_mom*(y+4.0))
psi /= sqrt(sum(conjugate(psi)*psi))

steps = 1000
substeps = int(1/dx)**3
psi_r = array(real(psi).flatten(), dtype='float64')
psi_i = array(imag(psi).flatten(), dtype='float64')

prob = f.create_dataset('prob', (steps, x.size, y.size), dtype='float64')
for i in range(steps):
    evolve(psi_r, psi_i, psi.shape[0], psi.shape[1],
           x[0, 0], y[0, 0], dx, dt/substeps,
           width, height, V0, 1, substeps)
    p = (psi_r**2 + psi_i**2).reshape(psi.shape)
    Z = sum(p)
    psi_r /= sqrt(Z)
    psi_i /= sqrt(Z)
    prob[i] = p[:]

f.close()
