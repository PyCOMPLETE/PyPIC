import numpy as np
from numpy.random import rand
from scipy.constants import epsilon_0
from numpy import pi

from xfields import TriLinearInterpolatedFieldMap

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
pweights = 1. + 0*x

# PIC
x_lim = (-1.1, 1.)
y_lim = (-1.2, 1.)
z_lim = (-1.3, 1.)

dx = 0.02
dy = 0.025
dz = 0.03

xg = np.arange(x_lim[0], x_lim[1]+0.1*dx, dx)

###############
# Actual test #
###############

# Build fieldmap object
fmap = TriLinearInterpolatedFieldMap(x_range=x_lim, dx=dx,
    y_range=y_lim, dy=dy, z_range=z_lim, dz=dz, solver='FFTSolver3D')

# Compute potential
fmap.update_from_particles(x_p=x, y_p=y, z_p=z, ncharges_p=pweights, q0=1.)

# Check on the x axis
rho_xg, phi_xg, ex_xg, _, _ = fmap.get_values_at_points(x=xg+center_xyz[0],
        y=0*xg+center_xyz[1], z=0*xg+center_xyz[2])


####################
# Plots and checks #
####################

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
int_rho = np.sum(fmap.rho)*dx*dy*dz
assert np.isclose(int_rho, len(x))


fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.pcolormesh(fmap.x_grid, fmap.y_grid, np.sum(fmap.rho, axis=2).T, shading='gouraud')
ax1.set_aspect('equal')
ax1.add_patch(plt.Circle((center_xyz[0], center_xyz[1]), radius,
                         color='w', fill=False))
ax1.set_xlabel('x')
ax1.set_ylabel('y')

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.pcolormesh(fmap.y_grid, fmap.z_grid, np.sum(fmap.rho, axis=0).T, shading='gouraud')
ax2.set_aspect('equal')
ax2.add_patch(plt.Circle((center_xyz[1], center_xyz[2]), radius,
                         color='w', fill=False))
ax2.set_xlabel('y')
ax2.set_ylabel('z')
plt.show()
