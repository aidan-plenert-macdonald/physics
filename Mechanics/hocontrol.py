import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

k = 1.0
eta = 1.0e-3
dt = 0.01
t = np.arange(0, 100, dt)

def z(t):
    return np.sin(t)

def r(t):
    return np.exp(-t/10)*r.rand[int(t/dt)]
r.rand = np.random.randn(t.size)

def step(w, t):
    T = np.array([[0, 1, 0, 0, 0],
                  [-k, 0, 0, 0, k],
                  [0, 0, 0, 1, 0],
                  [1, 0, -k, 0, 0],
                  [0, 0, -eta*k, 0, 0]])
    dw = (T.dot(w.reshape((-1, 1))) +
          np.array([[0, 0, 0, z(t), 0]]).T)
    return dw.flatten()

for x0 in np.arange(-1, -0.9, 0.1):
    W = odeint(step, [x0, 0, 0, 0, 0], t)
    plt.plot(t, z(t), 'b')
    plt.plot(t, W[:, 0], 'r')
    plt.plot(t, W[:, -1], 'g')
    plt.show()
