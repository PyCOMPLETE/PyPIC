import time

import numpy as np

# Pocl platform
from xfields.platforms.pocl import XfPoclPlatform
platform = XfPoclPlatform()

# # CPU platformr
# from xfields.platforms.cpu import XfCpuPlatform
# platform = XfCpuPlatform()

np2dev = platform.nparray_to_platform_mem
dev2np = platform.nparray_from_platform_mem

n_time = 1

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
data_gpu = platform.nparray_to_platform_mem(data_host)

fftobj = platform.plan_FFT(data_gpu, axes=(0,1,))

t1 = time.time()
for _ in range(n_time):
    fftobj.transform(data_gpu)
    transf_from_gpu = dev2np(data_gpu)

    fftobj.itransform(data_gpu)
    data_from_gpu = dev2np(data_gpu)
t2 = time.time()
print(f'time = {(t2-t1)/n_time:2e}')


import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(data[5,:,20])
plt.plot(np.real(data_from_gpu[5,:,20]))
plt.show()
