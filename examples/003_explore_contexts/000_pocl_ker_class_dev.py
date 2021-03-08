import numpy as np
import numpy.linalg as la

from xfields.platforms.pocl import XfPoclPlatform, XfPoclKernel

platform = XfPoclPlatform()

p2mk = platform.kernels.p2m_rectmesh3d

import pickle
with open('../000_sphere/picsphere.pkl', 'rb') as fid:
    ddd = pickle.load(fid)

fmap = ddd['fmap']
x0 = fmap.x_grid[0]
y0 = fmap.y_grid[0]
z0 = fmap.z_grid[0]

dx = fmap.dx
dy = fmap.dy
dz = fmap.dz

nx = fmap.nx
ny = fmap.ny
nz = fmap.nz


# Test p2m
n_gen = 1000000
x_gen_dev = platform.nparray_to_platform_mem(
        np.zeros([n_gen], dtype=np.float64)+fmap.x_grid[10]
        + 20* dx* np.linspace(0, 1., n_gen))
y_gen_dev = platform.nparray_to_platform_mem(
        np.zeros([n_gen], dtype=np.float64)+fmap.y_grid[10]
        + 20*dy* np.linspace(0, 1., n_gen))
z_gen_dev = platform.nparray_to_platform_mem(
        np.zeros([n_gen], dtype=np.float64)+fmap.z_grid[10]
        + 20*dz* np.linspace(0, 1., n_gen))
part_weights_dev = platform.nparray_to_platform_mem(
        np.arange(0, n_gen, 1,  dtype=np.float64))
dev_buff = platform.nparray_to_platform_mem(0*fmap._maps_buffer)
dev_rho = dev_buff[:,:,:,1] # This does not support .data
#dev_rho = dev_buff[:,:,:,0]

import time
t1 = time.time()
event = p2mk(nparticles=n_gen,
    x=x_gen_dev,
    y=y_gen_dev,
    z=z_gen_dev,
    part_weights=part_weights_dev,
    x0=x0, y0=y0, z0=z0, dx=dx, dy=dy, dz=dz,
    nx=nx, ny=ny, nz=nz,
    grid1d=dev_rho)
event.wait()
t2 = time.time()
print(f't = {t2-t1:.2e}')

assert(np.isclose(np.sum(dev_rho.get())*dx*dy*dz,
    np.sum(part_weights_dev.get())))
