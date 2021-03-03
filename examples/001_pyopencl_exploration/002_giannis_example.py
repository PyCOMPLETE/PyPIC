import numpy as np

import pyopencl as cl
import pyopencl.array as cla

import gpyfft

context = cl.create_some_context()
queue = cl.CommandQueue(context)

nn = 2048

t = np.linspace(0, 1, nn)
x = np.sin(2*np.pi*5*t)

data_host = np.zeros((nn, ), dtype = np.complex64)
data_host[:] = x
data_gpu = cla.to_device(queue, data_host)

fftobj = gpyfft.fft.FFT(context, queue, data_gpu, axes = (0,))

# gfft = gpyfft.GpyFFT(debug=False)
# plan = gfft.create_plan(context, data_gpu.shape)
# plan.bake(queue)

event1, = fftobj.enqueue_arrays(data_gpu)
event1.wait()
transf_from_gpu = data_gpu.get()

event2, = fftobj.enqueue_arrays(data_gpu, forward=False)
event2.wait()
x_from_gpu = data_gpu.get()

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.suptitle('Test round trip')
plt.plot(x)
plt.plot(np.real(x_from_gpu), '--')
plt.show()
