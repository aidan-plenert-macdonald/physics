"""
2D Free Fluid Simulation

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

q = a = np.array(np.meshgrid(np.arange(-5, 5, 0.01), np.arange(-5, 5, 0.01)))
dq = np.zeros(q.shape)
rho0 = 1.0*(np.abs(a[0, :, :]) < 0.5)*(np.abs(a[1, :, :]) < 0.5)
s = np.ones(rho0.shape)
dt = 1e-3

def div(X, Y):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( X, Y )
        c[ ~ np.isfinite( c )] = 0
    return c

def calc_J(q, a):
    dJ = np.array([[div(np.gradient(q[1, :, :], axis=0),
                        np.gradient(a[1, :, :], axis=0)),
                    div(-np.gradient(q[1, :, :], axis=0),
                        np.gradient(a[0, :, :], axis=1))],
                   [div(-np.gradient(q[0, :, :], axis=1),
                        np.gradient(a[1, :, :], axis=0)),
                    div(np.gradient(q[0, :, :], axis=1),
                        np.gradient(a[0, :, :], axis=1))]])
    J = dJ[0, 0, :, :]*dJ[1, 1, :, :] + dJ[1, 0, :, :]*dJ[0, 1, :, :]
    return J, dJ

def dP(rho0, J, dJ):
    """
    Compute the gradient of pressure

    I set the constants in U(rho, s) to 1 by choice of units
    """
    P = np.einsum('xy,kjxy->kjxy',
                  (1.0/3 - 1)*div(np.exp(s),
                                  div(rho0, J)**(2 - 1.0/3)),
                  dJ)
    DP = (div(np.gradient(P[:, 0, :, :], axis=1),
              np.gradient(a[0, :, :], axis=1)) +
          div(np.gradient(P[:, 1, :, :], axis=2),
              np.gradient(a[1, :, :], axis=0)))
    return DP

fig, ax = plt.subplots()
ax.contourf(a[0], a[1], rho0)

def update(frame_number):
    global q, dq
    J, dJ = calc_J(q, a)
    dq -= dt*div(dP(rho0, J, dJ),np.array([rho0]))
    q += dt*dq
    rho = div(rho0, J) # Wrong!!
    ax.clear()
    ax.contourf(q[0], q[1], rho)

animation = FuncAnimation(fig, update, interval=10,
                          frames=5000, blit=False)
plt.show()
