import sys
sys.path.append('../../xfields/csrc')
import p2m_cpu

import numpy as np
from numpy.random import rand
from scipy.constants import epsilon_0
from numpy import pi

from xfields import FFTSolver3D

import matplotlib.pyplot as plt
plt.close('all')

center_xyz = np.array([0.1, 0.2, -0.3])
radius = .5
n_part_cube = 10000000

x_cube = radius*(2. * rand(n_part_cube) - 1.) + center_xyz[0]
y_cube = radius*(2. * rand(n_part_cube) - 1.) + center_xyz[1]
z_cube = radius*(2. * rand(n_part_cube) - 1.) + center_xyz[2]

mask_sphere = ((x_cube - center_xyz[0])**2
             + (y_cube - center_xyz[1])**2
             + (z_cube - center_xyz[2])**2) < radius**2
x = x_cube[mask_sphere]
y = y_cube[mask_sphere]
z = z_cube[mask_sphere]

# PIC
x_lim = (-1.1, 1.)
y_lim = (-1.2, 1.)
z_lim = (-1.3, 1.)

dx = 0.02
dy = 0.025
dz = 0.03

xg = np.arange(x_lim[0], x_lim[1]+0.1*dx, dx)
yg = np.arange(y_lim[0], y_lim[1]+0.1*dy, dy)
zg = np.arange(z_lim[0], z_lim[1]+0.1*dz, dz)

nx = len(xg)
ny = len(yg)
nz = len(zg)

# Prepare arrays
rho = np.zeros((nx, ny, nz), dtype=np.float64, order='F')
phi = np.zeros((nx, ny, nz), dtype=np.float64, order='F')
dphi_dx = np.zeros((nx, ny, nz), dtype=np.float64, order='F')
dphi_dy = np.zeros((nx, ny, nz), dtype=np.float64, order='F')
dphi_dz = np.zeros((nx, ny, nz), dtype=np.float64, order='F')

# p2m
p2m_cpu.p2m(x, y, z, xg[0], yg[0], zg[0], dx, dy, dz, nx, ny, nz, rho)

# solve
solver = FFTSolver3D(dx=dx, dy=dy, dz=dz, nx=nx, ny=ny, nz=nz)
phi[:,:,:] = solver.solve(rho)

# Compute gradient
dphi_dx[1:nx-1,:,:] = 1/(2*dx)*(phi[2:,:,:]-phi[:-2,:,:])
dphi_dy[:,1:ny-1,:] = 1/(2*dy)*(phi[:,2:,:]-phi[:,:-2,:])
dphi_dz[:,:,1:nz-1] = 1/(2*dz)*(phi[:,:,2:]-phi[:,:,:-2])

# Interpolation
# Quick check on the x axis
rho_xg= np.zeros_like(xg)
phi_xg= np.zeros_like(xg)
ex_xg= np.zeros_like(xg)
p2m_cpu.m2p(xg+center_xyz[0],
        0*xg+center_xyz[1], 0*xg+center_xyz[2], xg[0], yg[0], zg[0],
        dx, dy, dz, nx, ny, nz,
        [rho, phi, dphi_dx],
        [rho_xg, phi_xg, ex_xg])
ex_xg *= (-1.)

plt.figure(100)
plt.plot(xg, rho_xg)
plt.axhline(y=len(x)/(4/3*np.pi*radius**3))

plt.figure(101)
plt.plot(xg, phi_xg)

e_ref = len(x)/(4*pi*epsilon_0) * (
        xg/radius**3*(np.abs(xg)<radius)
      + np.sign(xg)/xg**2*(np.abs(xg)>=radius))
plt.figure(102)
plt.plot(xg, ex_xg)
plt.plot(xg, e_ref)
plt.grid(True)

# Check integral
int_rho = np.sum(rho)*dx*dy*dz
assert np.isclose(int_rho, len(x))


fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.pcolormesh(xg, yg, np.sum(rho, axis=2).T, shading='gouraud')
ax1.set_aspect('equal')
ax1.add_patch(plt.Circle((center_xyz[0], center_xyz[1]), radius,
                         color='w', fill=False))
ax1.set_xlabel('x')
ax1.set_ylabel('y')

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.pcolormesh(yg, zg, np.sum(rho, axis=0).T, shading='gouraud')
ax2.set_aspect('equal')
ax2.add_patch(plt.Circle((center_xyz[1], center_xyz[2]), radius,
                         color='w', fill=False))
ax2.set_xlabel('y')
ax2.set_ylabel('z')
plt.show()
