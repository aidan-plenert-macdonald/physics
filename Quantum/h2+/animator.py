#
# Animator of simulation result
#  by Aidan Macdonald
#

from matplotlib.pyplot import *
from matplotlib.animation import FuncAnimation
from numpy import *
import h5py

dx = 0.05
X, Y = meshgrid(arange(-10, 10, dx), arange(-10, 10, dx))

f = h5py.File('transistor.h5')
prob = f['prob']

fig, ax = subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.grid()

def data(i):
    return ax.contourf(X, Y, prob[10*i])

ani = FuncAnimation(fig, data)

#show()
ani.save('transistor.mp4', writer='ffmpeg')
