import numpy as np
from numpy.random import rand
from scipy.constants import epsilon_0
from numpy import pi

import xfields.fieldmaps.linear_interpolators as li

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

gint_rep = np.zeros((2*nx, 2*ny, 2*nz), dtype=np.float64, order='F')
rho_rep = np.zeros((2*nx, 2*ny, 2*nz), dtype=np.float64, order='F')

# Build grid for primitive function
xg_F = np.arange(0, nx+2) * dx - dx/2
yg_F = np.arange(0, ny+2) * dy - dy/2
zg_F = np.arange(0, nz+2) * dz - dz/2
XX_F, YY_F, ZZ_F = np.meshgrid(xg_F, yg_F, zg_F, indexing='ij')

def primitive_func_3d(x,y,z):
    abs_r = np.sqrt(x * x + y * y + z * z)
    inv_abs_r = 1./abs_r
    res = 1./(4*pi*epsilon_0)*(
            -0.5 * (z*z * np.arctan(x*y*inv_abs_r/z)
                    + y*y * np.arctan(x*z*inv_abs_r/y)
                    + x*x * np.arctan(y*z*inv_abs_r/x))
               + y*z*np.log(x+abs_r)
               + x*z*np.log(y+abs_r)
               + x*y*np.log(z+abs_r))
    return res

# Compute primitive
F_temp = primitive_func_3d(XX_F, YY_F, ZZ_F)

# Integrated Green Function
gint_rep[:nx+1, :ny+1, :nz+1] = (F_temp[ 1:,  1:,  1:]
                               - F_temp[:-1,  1:,  1:]
                               - F_temp[ 1:, :-1,  1:]
                               + F_temp[:-1, :-1,  1:]
                               - F_temp[ 1:,  1:, :-1]
                               + F_temp[:-1,  1:, :-1]
                               + F_temp[ 1:, :-1, :-1]
                               - F_temp[:-1, :-1, :-1])

# Replicate
# To define how to make the replicas I have a look at:
# np.abs(np.fft.fftfreq(10))*10
# = [0., 1., 2., 3., 4., 5., 4., 3., 2., 1.]
gint_rep[nx+1:, :ny, :nz] = gint_rep[nx-1:0:-1, :ny, :nz]
gint_rep[:nx, ny+1:, :nz] = gint_rep[:nx, ny-1:0:-1, :nz]
gint_rep[nx+1:, ny+1:, :nz] = gint_rep[nx-1:0:-1, ny-1:0:-1, :nz]
gint_rep[:nx, :ny, nz+1:] = gint_rep[:nx, :ny, nz-1:0:-1]
gint_rep[nx+1:, :ny, nz+1:] = gint_rep[nx-1:0:-1,  :ny, nz-1:0:-1]
gint_rep[:nx, ny+1:, nz+1:] = gint_rep[:nx, ny-1:0:-1, nz-1:0:-1]
gint_rep[nx+1:, ny+1:, nz+1:] = gint_rep[nx-1:0:-1, ny-1:0:-1,nz:1:-1]

# Transform the green function
gint_rep_transf = np.fft.fftn(gint_rep)

# p2m
li.p2m(x, y, z, xg[0], yg[0], zg[0], dx, dy, dz, nx, ny, nz, rho)

# solve
rho_rep[:,:,:] = 0. # reset
rho_rep[:nx, :ny, :nz] = rho
phi_rep = np.fft.ifftn(np.fft.fftn(rho_rep) * gint_rep_transf)
assert np.max(np.abs(np.imag(phi_rep)))/np.max(np.abs(rho_rep))<1e-5
phi[:,:,:] = np.real(phi_rep)[:nx, :ny, :nz]

# Compute gradient
dphi_dx[1:nx-1,:,:] = 1/(2*dx)*(phi[2:,:,:]-phi[:-2,:,:])
dphi_dy[:,1:ny-1,:] = 1/(2*dy)*(phi[:,2:,:]-phi[:,:-2,:])
dphi_dz[:,:,1:nz-1] = 1/(2*dz)*(phi[:,:,2:]-phi[:,:,:-2])

# Quick check on the x axis - rho
res = np.zeros_like(xg)
li.m2p(xg, 0*xg, 0*xg, xg[0], yg[0], zg[0],
        dx, dy, dz, nx, ny, nz, [rho], [res])
plt.figure(100)
plt.plot(xg, res)
plt.axhline(y=len(x)/(4/3*np.pi*radius**3))

# Quick check on the x axis - phi
res = np.zeros_like(xg)
li.m2p(xg, 0*xg, 0*xg, xg[0], yg[0], zg[0],
        dx, dy, dz, nx, ny, nz, [phi], [res])
plt.figure(101)
plt.plot(xg, res)

# Quick check on the x axis - phi
ex= np.zeros_like(xg)
li.m2p(xg+center_xyz[0],
        0*xg+center_xyz[1], 0*xg+center_xyz[2], xg[0], yg[0], zg[0],
        dx, dy, dz, nx, ny, nz, [dphi_dx], [ex])
ex *= (-1.)
e_ref = len(x)/(4*pi*epsilon_0) * (
        xg/radius**3*(np.abs(xg)<radius)
      + np.sign(xg)/xg**2*(np.abs(xg)>=radius))
plt.figure(102)
plt.plot(xg, ex)
plt.plot(xg, e_ref)
plt.grid(True)

plt.figure(101)
plt.plot(xg, res)
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
