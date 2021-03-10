import numpy as np

from pysixtrack.particles import Particles

from xfields.platforms import XfCpuPlatform
platform = XfCpuPlatform()

#from xfields.platforms import XfCupyPlatform
#platform = XfCupyPlatform(default_block_size=256)

from xfields import SpaceCharge3D

print(repr(platform))

class CupyMathlib(object):

    from cupy import sqrt, exp, sin, cos, abs, pi, tan

    @classmethod
    def wfun(cls, z_re, z_im):
        raise NotImplementedError

mathlib = CupyMathlib()


n_macroparticles = int(1e6)
bunch_intensity = 2.5e11
sigma_x = 3e-3
sigma_y = 2e-3
sigma_z = 30e-2
p0c = 25.92e9
mass = Particles.pmass,

x_part = sigma_x * np.random.normal(size=(n_macroparticles,))
y_part = sigma_y * np.random.normal(size=(n_macroparticles,))
z_part = sigma_z * np.random.normal(size=(n_macroparticles,))
weights_part = 0*x_part + bunch_intensity/n_macroparticles

# insert probes
n_probes = 1000
r_probes = np.linspace(-2e-2, 2e-2, n_probes)
theta_probes = 0.
x_probes = r_probes * np.cos(theta_probes)
y_probes = r_probes * np.sin(theta_probes)
z_probes = 0 * x_probes

x_part = np.concatenate([x_probes, x_part])
y_part = np.concatenate([y_probes, y_part])
z_part = np.concatenate([z_probes, z_part])
weights_part = np.concatenate([0*x_probes, weights_part])

px_part = 0*x_part
py_part = 0*x_part
pt_part = 0*x_part

# Move to platform
np2platf = platform.nparray_to_platform_mem
x_part_dev = np2platf(x_part)
y_part_dev = np2platf(y_part)
z_part_dev = np2platf(z_part)
px_part_dev = np2platf(px_part)
py_part_dev = np2platf(py_part)
ptau_part_dev = np2platf(pt_part)
weights_part_dev = np2platf(weights_part)

particles = Particles(
        p0c=p0c,
        mass = mass,
        mathlib=mathlib,
        x=x_part_dev,
        y=y_part_dev,
        zeta=z_part_dev,
        px=px_part_dev,
        py=py_part_dev,
        ptau=ptau_part_dev)
particles.weight = weights_part_dev


x_range = 5.*sigma_x*np.array([-1, 1])
y_range = 5.*sigma_y*np.array([-1, 1])
z_range = 5.*sigma_z*np.array([-1, 1])

nx = 256
ny = 256
nz = 50

spcharge = SpaceCharge3D(
        length=1, update_on_track=True, apply_z_kick=True,
        x_range=x_range, y_range=y_range, z_range=z_range,
        nx=nx, ny=ny, nz=nz,
        solver='FFTSolver2p5D',
        platform=platform)

spcharge.track(particles)

p2np = platform.nparray_from_platform_mem

import matplotlib.pyplot as plt
plt.close('all')
plt.figure()
plt.plot(r_probes, p2np(particles.px[:n_probes]), '.-')
plt.plot(r_probes, p2np(particles.py[:n_probes]))
plt.show()

