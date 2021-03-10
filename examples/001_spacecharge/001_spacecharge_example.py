import numpy as np

from pysixtrack.particles import Particles

from xfields.platforms import XfCpuPlatform, XfCupyPlatform
###################
# Choose platform #
###################

platform = XfCpuPlatform()
#platform = XfCupyPlatform(default_block_size=256)

print(repr(platform))

n_macroparticles = int(1e6)
bunch_intensity = 2.5e11
sigma_x = 3e-3
sigma_y = 2e-3
sigma_z = 30e-2
p0c = 25.92e9
mass = Particles.pmass,
theta_probes = 0.
r_max_probes = 2e-2
n_probes = 1000

from temp_makepart import generate_particles_object
particles, r_probes = generate_particles_object(platform,
                            n_macroparticles,
                            bunch_intensity,
                            sigma_x,
                            sigma_y,
                            sigma_z,
                            p0c,
                            mass,
                            n_probes,
                            r_max_probes,
                            theta_probes)

x_range = 5.*sigma_x*np.array([-1, 1])
y_range = 5.*sigma_y*np.array([-1, 1])
z_range = 5.*sigma_z*np.array([-1, 1])

nx = 256
ny = 256
nz = 50

from xfields import SpaceCharge3D

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

