"""
2D Free Fluid Simulation

"""
import numpy as np


q = a = np.array(np.meshgrid(arange(-5, 5, 0.01), np.arange(-5, 5, 0.01)))
rho0 = 1.0*(np.abs(a[0, :, :]) < 0.5)*(np.abs(a[1, :, :]) < 0.5)
s = np.ones(rho0.shape)

def calc_J(q, a):
    dJ = np.array([[diff(q[1, :, :], axis=1)/diff(a[1, :, :], axis=1),
                    -diff(q[1, :, :], axis=0)/diff(a[0, :, :], axis=0)],
                   [-diff(q[0, :, :], axis=1)/diff(a[1, :, :], axis=1),
                    diff(q[0, :, :], axis=0)/diff(a[0, :, :], axis=0)]])
    J = dJ[0, 0, :, :]*dJ[1, 1, :, :] + dJ[1, 0, :, :]*dJ[0, 1, :, :]
    return J, dJ


def dP(rho0, J, dJ):
    """
    Compute the gradient of pressure

    I set the constants in U(rho, s) to 1 by choice of units
    """
    P = np.einsum('xy,kjxy->kjxy', (1.0/3 - 1)*(rho0/J)**(1.0/3 - 2)*np.exp(s), dJ))
    return 

