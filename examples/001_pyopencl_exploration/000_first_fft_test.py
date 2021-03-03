import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from gpyfft.fft import FFT

context = cl.create_some_context()
queue = cl.CommandQueue(context)

nn = 1024

t = np.linspace(0, 1, nn)
x = np.sin(2*np.pi*5*t)

data_host = np.zeros((nn, ), dtype = np.complex64)
data_host[:] = x
data_gpu = cla.to_device(queue, data_host)

transform = FFT(context, queue, data_gpu, axes = (0,))

event, = transform.enqueue()
event.wait()

result_host = data_gpu.get()
