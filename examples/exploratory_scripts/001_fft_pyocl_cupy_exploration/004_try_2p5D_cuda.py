import time

import numpy as np

import cupy as cp

from cupyx.scipy import fftpack as cufftp

n_time = 3

nn_x = 256*2
nn_y = 256*2
nn_z = 50

x = np.linspace(0, 1, nn_x)
y = np.linspace(0, 1, nn_y)
z = np.linspace(0, 1, nn_z)

XX_F, YY_F, ZZ_F = np.meshgrid(x, y, z, indexing='ij')
data = np.sin(2*np.pi*(50-20*(1-ZZ_F))*XX_F)*np.cos(2*np.pi*70*YY_F)

data_host = np.zeros((nn_x, nn_y, nn_z), dtype = np.complex64, order='F')
data_from_gpu = np.zeros((nn_x, nn_y, nn_z),
                           dtype = np.complex64, order='F')
data_host[:] = data

data_gpu = cp.array(data_host)

plan = cufftp.get_fft_plan(data_host, axes=(0,1), value_type='C2C')

t1 = time.time()
for _ in range(n_time):
    transf_gpu = cufftp.fftn(data_gpu, axes=(0, 1), plan=plan)
    res_gpu = cufftp.ifftn(transf_gpu, axes=(0, 1), plan=plan)
    data_from_gpu = res_gpu.get()
t2 = time.time()
print(f't_gpu = {(t2-t1)/n_time:2e}')


_ = np.fft.ifftn(np.fft.fftn(data_host, axes=(0,1)), axes=(0,1))
t1 = time.time()
for _ in range(n_time):
    temp = np.fft.ifftn(np.fft.fftn(data_host, axes=(0,1)), axes=(0,1))
t2 = time.time()
print(f't_numpy = {(t2-t1)/n_time:2e}')

import pyfftw
for n_threads in [1, 4]:
    fftw = pyfftw.builders.fftn(data_host, axes=(0,1), threads=n_threads)
    ifftw = pyfftw.builders.ifftn(data_host, axes=(0,1), threads=n_threads)
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
