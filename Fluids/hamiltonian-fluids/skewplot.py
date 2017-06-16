"""
Checking to see if we can skew 3D plots by skewing the X, Y
coordinates
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X, Y = np.meshgrid(np.arange(-3, 3, 0.01), np.arange(-3, 3, 0.01))
Z = np.exp(-(X**2 + Y**2/4)/2)

R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X) + 1.0e0*R
X, Y = R*np.cos(Theta), R*np.sin(Theta)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()
