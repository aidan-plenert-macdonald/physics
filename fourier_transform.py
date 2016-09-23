#
# "Continous Fourier Transform"
# 
# A means of computing the fourier
# and inverse fourier transform
# of a function with effectively
# finite support
#
# By using the Fast Fourier Transform,
# provided with Numpy and shift properties,
# we can compute an plot
# the transform, and reconstruction
#
# The inverse transform uses a fourier
# transform property to use prior code
# by simply reversing the transform
# frequencies
#
from numpy import *
from matplotlib.pyplot import *

#
# How to make use of the Numpy fft
#
def fourier_transform(f, t):
    dt = t[1] - t[0]
    t0 = min(t)
    F = fft.fft(f)
    w = fft.fftfreq(t.size)*2*pi/dt
    # Shift and normalization
    F *= dt*exp(-(1j)*w*t0)/sqrt(2*pi)
    data = zip(w, F)
    data.sort()
    w, F = zip(*data)
    return array(F), array(w)

def inv_fourier_transform(F, w):
    f, t = fourier_transform(F, w)
    return f, array([x for x in reversed(t)])

# Transform of a gaussian
# which is a still point in the F. Transform
# for simplicity of demonstration
t = arange(-10, 10, 0.01)
f = exp(-t**2/2)/sqrt(2*pi)
subplot(3, 1, 1)
plot(t, f)
title("Original function")

F, w = fourier_transform(f, t)
subplot(3, 1, 2)
plot(w, F)
title("Fourier Transform")

fp, tp = inv_fourier_transform(F, w)
subplot(3, 1, 3)
plot(tp, fp)
title("Reconstructed")

show()
