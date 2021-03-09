import time

import numpy as np

import pyopencl as cl
import pyopencl.array as cla

import gpyfft

context = cl.create_some_context()
queue = cl.CommandQueue(context)

n_time = 1

nn_x = 256
nn_y = 512
nn_z = 100

x = np.linspace(0, 1, nn_x)
y = np.linspace(0, 1, nn_y)
z = np.linspace(0, 1, nn_z)

XX_F, YY_F, ZZ_F = np.meshgrid(x, y, z, indexing='ij')
data = np.sin(2*np.pi*(50-20*(1-ZZ_F))*XX_F)*np.cos(2*np.pi*70*YY_F)

data_host = np.zeros((nn_x, nn_y, nn_z), dtype = np.complex128, order='F')
data_host[:] = data
data_gpu = cla.to_device(queue, data_host)

fftobj = gpyfft.fft.FFT(context, queue, data_gpu, axes = (0,1,2))

t1 = time.time()
for _ in range(n_time):
    event1, = fftobj.enqueue_arrays(data_gpu)
    event1.wait()
    transf_from_gpu = data_gpu.get()

    event2, = fftobj.enqueue_arrays(data_gpu, forward=False)
    event2.wait()
    data_from_gpu = data_gpu.get()
t2 = time.time()
print(f't_gpu = {(t2-t1)/n_time:2e}')


_ = np.fft.ifftn(np.fft.fftn(data_host, axes=(0,1)), axes=(0,1,2))
t1 = time.time()
for _ in range(n_time):
    temp = np.fft.ifftn(np.fft.fftn(data_host, axes=(0,1,2)), axes=(0,1,2))
t2 = time.time()
print(f't_numpy = {(t2-t1)/n_time:2e}')

import pyfftw
for n_threads in [1, 4]:
    fftw = pyfftw.builders.fftn(data_host, axes=(0,1,2), threads=n_threads)
    ifftw = pyfftw.builders.ifftn(data_host, axes=(0,1,2), threads=n_threads)
    t1 = time.time()
    for _ in range(n_time):
        temp = ifftw(fftw(data_host))
    t2 = time.time()
    print(f't_fftw = {(t2-t1)/n_time:2e} ({n_threads} threads)')

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(data[5,:,20])
plt.plot(np.real(data_from_gpu[5,:,20]))
plt.show()
