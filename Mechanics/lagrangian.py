#
# General Simulation for Any Lagrangian!!
#   by Aidan Macdonald
# 
# Uses Numpy to take derivatives, but written
# to support closed form derivatives
# 

import numpy, sympy
from scipy.optimize import root
from numpy import *

eps = 1.0e-4

# Double Pendulum Lagrangian
# See Landau Mechanics Ch. 1 Problem 1
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0
def L(q, dq):
    T = numpy.array([(m1 + m2)*l1**2/2, m2*l2**2/2]).dot(dq) + m2*l1*l2*dq[0]*dq[1]*cos(q[0] - q[1])
    V = -(m1 + m2)*l1*cos(q[0]) - m2*l2*cos(q[1])
    return T - V

def grad(f):
    return lambda x: numpy.array([(f(x + eps*numpy.eye(x.size)[i]) -
                                   f(x - eps*numpy.eye(x.size)[i]))/(2*eps)
                                  for i in range(x.size)])

def DLdq(q, dq):
    return grad(lambda _dq: L(q, _dq))(dq)

def DLq(q, dq):
    return grad(lambda _q: L(_q, dq))(q)


def take_step(T, q, dq):
    for t in numpy.arange(0, T, eps):
        _DLdq = DLdq(q, dq)
        _DLq  = DLq(q,  dq)

        new_dq = root(lambda _dq:
                      (DLdq(q, _dq) -
                      _DLdq - eps*_DLq).flatten(),
                      dq, tol=eps).x
        q  = q + (dq + new_dq)*eps/2.0
        dq = new_dq
    return q, dq


from matplotlib.pyplot import *
from matplotlib import animation

fig, ax = subplots()
line,   = ax.plot([], [], lw=2)
ax.set_ylim(-3, 3)
ax.set_xlim(-3, 3)
ax.grid()
title('Double Pendulum')

def data_gen():
    dq = numpy.array([0, 0])
    q  = numpy.array([numpy.pi/4, numpy.pi/2])
    
    for i in xrange(2000):
        q, dq = take_step(0.001, q, dq)
        x = [0, l1*sin(q[0]), l1*sin(q[0]) + l2*sin(q[1])]
        y = [0, -l1*cos(q[0]), -l1*cos(q[0]) - l2*cos(q[1])]
        yield x, y

def run((x, y)):
    line.set_data(x, y)
    return line,

ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=1)
#ani.save('double-pendulum-lagrangian.mp4', writer='ffmpeg')
show()        


