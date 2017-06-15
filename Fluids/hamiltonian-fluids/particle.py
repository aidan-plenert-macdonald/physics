"""
2D Many Particle Simulation

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

N_PARTICLES = int(1e6)

upper_bnds =  np.array([[5, 5]])
lower_bnds = -np.array([[5, 5]])

X = np.random.rand(N_PARTICLES, 2) - 0.5
P = np.random.randn(N_PARTICLES, 2)
dt = 1e-3
m = 1.0

fig, ax = plt.subplots()
ax.set_xlim(1.2*lower_bnds[0, 0], 1.2*upper_bnds[0, 0])
ax.set_ylim(1.2*lower_bnds[0, 1], 1.2*upper_bnds[0, 0])

scat = ax.scatter(X[:, 0], X[:, 1], s=np.pi, c='r', alpha=0.1)

def update(frame_number):
    global X, P
    X += dt*P/m
    P += -2*P*(1.0*(X < lower_bnds) + 1.0*(X > upper_bnds))
    scat.set_offsets(X)

writer = FFMpegWriter(fps=15, bitrate=5000)
animation = FuncAnimation(fig, update, interval=10, frames=5000)
animation.save('particles.mp4', writer=writer, dpi=600)
#plt.show()
