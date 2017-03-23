from numpy import *
from matplotlib.pyplot import *

x = arange(0, 10, 0.01).reshape(-1, 1)
y = arange(0, 10, 0.01).reshape(1, -1)
d = 0.1*x + 0.0*y + 1

g = 9.8

def k(omega, eps=1e-4, maxiter=1000):
    """
    C = omega/k = sqrt(g/k * tanh(k*d))
    
    We need to find the root of,
    f(k)  = omega/k - sqrt(g/k * tanh(k*d))
    From Sympy,
    f'(k) = (k*sqrt(g*tanh(d*k)/k)*(d*k*(tanh(d*k)**2 - 1) + tanh(d*k))/2 - omega*tanh(d*k))/(k**2*tanh(d*k))

    which is convex. k should naturally be small maybe
    around k = 2*pi/20 (ie. long wave length of 20m)
    
    Newton-Raphson method,
    k[n+1] = k[n] - f(k[n])/f'(k[n])
    """
    k = 2*pi/20 * ones(d.shape)

    f = lambda k: omega/k - sqrt(g/k * tanh(k*d))
    Df = lambda k: (k*sqrt(g*tanh(d*k)/k)*(d*k*(tanh(d*k)**2 - 1) + tanh(d*k))/2 - omega*tanh(d*k))/(k**2*tanh(d*k))

    for i in range(maxiter):
        k = k - f(k)/Df(k)
        err = linalg.norm(f(k))/d.size
        if linalg.norm(f(k))/d.size <= eps:
            break
        elif i % 20 == 0 and i > 0:
            print "Round", i, "error", err
    return k

omega = 2*pi/12
k = k(omega)

X, Y = meshgrid(x, y)
contour(X, Y, sin(k*X))
show()
    
    
    
