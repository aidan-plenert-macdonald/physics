import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

k = 1.0
eta = 5.0e-3

def z(t):
    return np.sin(t)

def step(w, t):
    T = np.array([[0, 1, 0, 0, 0],
                  [-k, 0, 0, 0, k],
                  [0, 0, 0, 1, 0],
                  [1, 0, -k, 0, 0],
                  [0, 0, -eta*k, 0, 0]])
    dw = (T.dot(w.reshape((-1, 1))) +
          np.array([[0, 0, 0, z(t), 0]]).T)
    return dw.flatten()

t = np.arange(0, 50, 0.01)
W = odeint(step, [0.7, 0, 0, 0, 0], t)

plt.plot(t, z(t), 'b')
plt.plot(t, W[:, 0], 'r')
plt.show()
