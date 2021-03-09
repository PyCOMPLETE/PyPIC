import numpy as np
from numpy.random import rand
from scipy.constants import epsilon_0
from numpy import pi

from xfields import TriLinearInterpolatedFieldMap

from xfields.platforms import XfCpuPlatform
platform = XfCpuPlatform()

from xfields.platforms import XfCupyPlatform
platform = XfCupyPlatform(default_block_size=256)

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

nx = 128
ny = 64
nz = 50

theta_test = 70. * np.pi/360
phi_test = 300 * np.pi/360.
r_test = 2.5*radius*np.linspace(-1, 1., 1000)

x0_test = r_test * np.sin(theta_test) * np.cos(phi_test)
y0_test = r_test * np.sin(theta_test) * np.sin(phi_test)
z0_test = r_test * np.cos(theta_test)

x_test = center_xyz[0] + x0_test
y_test = center_xyz[1] + y0_test
z_test = center_xyz[2] + z0_test

# Moving particles and test coordinates to GPU (if needed)
np2platf = platform.nparray_to_platform_mem
x_dev = np2platf(x)
y_dev = np2platf(y)
z_dev = np2platf(z)
pweights_dev = np2platf(pweights)
x_test_dev = np2platf(x_test)
y_test_dev = np2platf(y_test)
z_test_dev = np2platf(z_test)

###############
# Actual test #
###############

# Build fieldmap object
fmap = TriLinearInterpolatedFieldMap(x_range=x_lim, nx=nx,
    y_range=y_lim, ny=ny, z_range=z_lim, nz=nz, solver='FFTSolver3D',
    platform=platform)


# Compute potential
fmap.update_from_particles(x_p=x_dev, y_p=y_dev, z_p=z_dev,
        ncharges_p=pweights_dev, q0=1.)

# Check on the x axis
(rho_test_dev, phi_test_dev, dx_test_dev, dy_test_dev,
        dz_test_dev) = fmap.get_values_at_points(
            x=x_test_dev, y=y_test_dev, z=z_test_dev)

# Copy back for plotting
platf2np = platform.nparray_from_platform_mem
rho_test =  platf2np(rho_test_dev)
phi_test = platf2np(phi_test_dev)
dx_test = platf2np(dx_test_dev)
dy_test = platf2np(dy_test_dev)
dz_test = platf2np(dz_test_dev)

# pickle for other tests
import pickle
# with open('picsphere.pkl', 'wb') as fid:
#     pickle.dump({
#         'fmap': fmap,
#         'x_test': x_test,
#         'y_test': y_test,
#         'z_test': z_test,
#         }, fid)

####################
# Plots and checks #
####################

ex_test = -dx_test
ey_test = -dy_test
ez_test = -dz_test

plt.figure(100)
plt.plot(r_test, rho_test)
plt.axhline(y=len(x)/(4/3*np.pi*radius**3))

plt.figure(101)
plt.plot(r_test, phi_test)

e_ref = len(x)/(4*pi*epsilon_0) * (
        r_test/radius**3*(np.abs(r_test)<radius)
      + np.sign(r_test)/r_test**2*(np.abs(r_test)>=radius))
plt.figure(102)
plt.plot(r_test, ex_test*x0_test/r_test
        + ey_test*y0_test/r_test + ez_test*z0_test/r_test)
plt.plot(r_test, e_ref)
plt.grid(True)

# Check integral
int_rho = np.sum(fmap.rho)*fmap.dx*fmap.dy*fmap.dz
assert np.isclose(int_rho, len(x))


fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.pcolormesh(fmap.x_grid, fmap.y_grid,
        np.sum(platf2np(fmap.rho), axis=2).T, shading='gouraud')
ax1.set_aspect('equal')
ax1.add_patch(plt.Circle((center_xyz[0], center_xyz[1]), radius,
                         color='w', fill=False))
ax1.set_xlabel('x')
ax1.set_ylabel('y')

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.pcolormesh(fmap.y_grid, fmap.z_grid,
        np.sum(platf2np(fmap.rho), axis=0).T, shading='gouraud')
ax2.set_aspect('equal')
ax2.add_patch(plt.Circle((center_xyz[1], center_xyz[2]), radius,
                         color='w', fill=False))
ax2.set_xlabel('y')
ax2.set_ylabel('z')
plt.show()
