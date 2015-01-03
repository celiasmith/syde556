import syde556
import numpy

T = 4.0         # length of signal in seconds
dt = 0.001      # time step size

# Generate bandlimited white noise (use your own function from part 1.1)
x, X = syde556.generate_signal(T, dt, rms=0.5, limit=5, seed=3)

Nt = len(x)                #
t = numpy.arange(Nt) * dt  #

# Neuron parameters
tau_ref = 0.002          #
tau_rc = 0.02            #
x0 = 0.0                 # firing rate at x=x0 is a0
a0 = 40.0
x1 = 1.0                 # firing rate at x=x1 is a1
a1 = 150.0

# 
eps = tau_rc/tau_ref
r1 = 1.0 / (tau_ref * a0)
r2 = 1.0 / (tau_ref * a1)
f1 = (r1 - 1) / eps
f2 = (r2 - 1) / eps
alpha = (1.0/(numpy.exp(f2)-1) - 1.0/(numpy.exp(f1)-1))/(x1-x0) 
x_threshold = x0-1/(alpha*(numpy.exp(f1)-1))              
Jbias = 1-alpha*x_threshold;   

# Simulate the two neurons (use your own function from part 3)
spikes = syde556.two_neurons(x, dt, alpha, Jbias, tau_rc, tau_ref)


freq = numpy.arange(Nt)/T - Nt/(2.0*T)   #
omega = freq*2*numpy.pi                  #

r = spikes[0] - spikes[1]                #
R = numpy.fft.fftshift(numpy.fft.fft(r)) #


sigma_t = 0.025                          #
W2 = numpy.exp(-omega**2*sigma_t**2)     #
W2 = W2 / sum(W2)                        #

CP = X*R.conjugate()                  #
WCP = numpy.convolve(CP, W2, 'same')  #
RP = R*R.conjugate()                  #
WRP = numpy.convolve(RP, W2, 'same')  #
XP = X*X.conjugate()                  #
WXP = numpy.convolve(XP, W2, 'same')  #

H = WCP / WRP                         #

h = numpy.fft.fftshift(numpy.fft.ifft(numpy.fft.ifftshift(H))).real  #

XHAT = H*R                            #

xhat = numpy.fft.ifft(numpy.fft.ifftshift(XHAT)).real  #


import pylab

pylab.figure(1)
pylab.subplot(1,2,1)
pylab.plot(freq, numpy.sqrt(XP), label='???')  # 
pylab.legend()
pylab.xlabel('???')
pylab.ylabel('???')

pylab.subplot(1,2,2)
pylab.plot(freq, numpy.sqrt(RP), label='???')  # 
pylab.legend()
pylab.xlabel('???')
pylab.ylabel('???')


pylab.figure(2)
pylab.subplot(1,2,1)
pylab.plot(freq, H.real)   #
pylab.xlabel('???')
pylab.title('???')
pylab.xlim(-50, 50)

pylab.subplot(1,2,2)
pylab.plot(t-T/2, h)       #
pylab.title('???')
pylab.xlabel('???')
pylab.xlim(-0.5, 0.5)


pylab.figure(3)
pylab.plot(t, r, color='k', label='???', alpha=0.2)  #
pylab.plot(t, x, linewidth=2, label='???')           #
pylab.plot(t, xhat, label='???')                     #
pylab.title('???')
pylab.legend(loc='best')
pylab.xlabel('???')

pylab.show()
