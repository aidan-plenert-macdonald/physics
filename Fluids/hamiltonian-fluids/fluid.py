"""
2D Free Fluid Simulation

"""
import numpy as np


q = a = np.array(np.meshgrid(arange(-5, 5, 0.01), np.arange(-5, 5, 0.01)))
dq = np.zeros(q.shape)
rho0 = 1.0*(np.abs(a[0, :, :]) < 0.5)*(np.abs(a[1, :, :]) < 0.5)
s = np.ones(rho0.shape)
dt = 1e-3

def calc_J(q, a):
    dJ = np.array([[np.gradient(q[1, :, :], axis=1)/np.gradient(a[1, :, :], axis=1),
                    -np.gradient(q[1, :, :], axis=0)/np.gradient(a[0, :, :], axis=0)],
                   [-np.gradient(q[0, :, :], axis=1)/np.gradient(a[1, :, :], axis=1),
                    np.gradient(q[0, :, :], axis=0)/np.gradient(a[0, :, :], axis=0)]])
    J = dJ[0, 0, :, :]*dJ[1, 1, :, :] + dJ[1, 0, :, :]*dJ[0, 1, :, :]
    return J, dJ

def dP(rho0, J, dJ):
    """
    Compute the gradient of pressure

    I set the constants in U(rho, s) to 1 by choice of units
    """
    P = np.einsum('xy,kjxy->kjxy', (1.0/3 - 1)*(rho0/J)**(1.0/3 - 2)*np.exp(s), dJ))
    return (np.gradient(P[:, 0, :, :], axis=1)/np.gradient(a[0, :, :], axis=0) +
            np.gradient(P[:, 1, :, :], axis=2)/np.gradient(a[1, :, :], axis=1))

def update():
    global q, dq
    J, dJ = calc_J(q, a)
    dq -= dt*dP(rho0, J, dJ)/np.array([rho0])
    q += dt*dq 
    rho = rho0/J # Wrong!!
    return rho
    
