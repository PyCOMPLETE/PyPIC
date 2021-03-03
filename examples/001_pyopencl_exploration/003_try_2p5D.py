import numpy as np

import pyopencl as cl
import pyopencl.array as cla

import gpyfft

context = cl.create_some_context()
queue = cl.CommandQueue(context)

nn_x = 256
nn_y = 512
nn_z = 100

x = np.linspace(0, 1, nn_x)
y = np.linspace(0, 1, nn_y)
z = np.linspace(0, 1, nn_z)

XX_F, YY_F, ZZ_F = np.meshgrid(x, y, z, indexing='ij')
data = np.sin(2*np.pi*(50-20*(1-ZZ_F))*XX_F)*np.cos(2*np.pi*70*YY_F)

data_host = np.zeros((nn_x, nn_y, nn_z), dtype = np.complex64, order='F')
data_host[:] = data
data_gpu = cla.to_device(queue, data_host)

fftobj = gpyfft.fft.FFT(context, queue, data_gpu, axes = (0,1,))

event1, = fftobj.enqueue_arrays(data_gpu)
event1.wait()
transf_from_gpu = data_gpu.get()

event2, = fftobj.enqueue_arrays(data_gpu, forward=False)
event2.wait()
x_from_gpu = data_gpu.get()

