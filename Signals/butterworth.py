#
# Butterworth Filter Example
#
#

from numpy import *
from matplotlib.pyplot import *
from scipy import signal

t = arange(0, 1, 1.0e-4)
y = sin(2*pi*100*t) + sin(2*pi*200*t)
f_len = 10

b, a = signal.butter(8, 150, 'lowpass', analog=True)

f = fft.fftfreq(t.size, d=1.0e-4)
H = abs(polyval(b, 1j*2*pi*f)/polyval(a, 1j*2*pi*f))
Y = H*abs(fft.fft(y)/t.size)

semilogy(f[abs(f) <= 300], Y[abs(f) <= 300])
show()
