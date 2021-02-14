import sys
sys.path.append('../../xfields/csrc')

import numpy as np
from numpy.random import rand

import p2m_cpu
import matplotlib.pyplot as plt

center_xyz = np.array([0.1, 0.2, -0.3])
radius = .5
n_part_cube = 10000

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
x_lim = (-1., 1.)
y_lim = (-1., 1.)
z_lim = (-1., 1.)

dx = 0.01
dy = 0.015
dz = 0.012

xg = np.arange(x_lim[0], x_lim[1]+0.1*dx, dx)
yg = np.arange(y_lim[0], y_lim[1]+0.1*dy, dy)
zg = np.arange(z_lim[0], z_lim[1]+0.1*dz, dz)

nx = len(xg)
ny = len(yg)
nz = len(zg)

rho = np.zeros((nx, ny, nz), dtype=np.float64, order='F')

p2m_cpu.p2m(x, y, z, xg[0], yg[0], zg[0], dx, dy, dz, nx, ny, nz, rho)

# Check integral
int_rho = np.sum(rho)*dx*dy*dz
assert np.isclose(int_rho, len(x))

plt.close('all')
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
