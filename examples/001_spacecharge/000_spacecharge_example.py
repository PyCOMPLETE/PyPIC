import time

import numpy as np

from pysixtrack.particles import Particles
from xfields.platforms import XfCpuPlatform, XfCupyPlatform

###################
# Choose platform #
###################

platform = XfCpuPlatform()
platform = XfCupyPlatform(default_block_size=256)

print(repr(platform))

#################################
# Generate particles and probes #
#################################

n_macroparticles = int(1e6)
bunch_intensity = 2.5e11
sigma_x = 3e-3
sigma_y = 2e-3
sigma_z = 3e-3
p0c = 25.92e9
mass = Particles.pmass,
theta_probes = 30 * np.pi/180
r_max_probes = 2e-2
z_probes = 1.2*sigma_z
n_probes = 1000

from temp_makepart import generate_particles_object
(particles, r_probes, x_probes,
        y_probes, z_probes) = generate_particles_object(platform,
                            n_macroparticles,
                            bunch_intensity,
                            sigma_x,
                            sigma_y,
                            sigma_z,
                            p0c,
                            mass,
                            n_probes,
                            r_max_probes,
                            z_probes,
                            theta_probes)

######################
# Space charge (PIC) #
######################

x_lim = 5.*sigma_x
y_lim = 5.*sigma_y
z_lim = 5.*sigma_z

from xfields import SpaceCharge3D

spcharge = SpaceCharge3D(
        length=1, update_on_track=True, apply_z_kick=False,
        x_range=(-x_lim, x_lim),
        y_range=(-y_lim, y_lim),
        z_range=(-z_lim, z_lim),
        nx=256, ny=256, nz=50,
        solver='FFTSolver3D',
        gamma0=particles.gamma0,
        platform=platform)

spcharge.track(particles)


##############################
# Compare against pysixtrack #
##############################


p2np = platform.nparray_from_platform_mem

from pysixtrack.elements import SpaceChargeBunched
scpyst = SpaceChargeBunched(
        number_of_particles = bunch_intensity,
        bunchlength_rms=sigma_z,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        length=spcharge.length,
        x_co=0.,
        y_co=0.)

p_pyst = Particles(p0c=p0c,
        mass=mass,
        x=x_probes.copy(),
        y=y_probes.copy(),
        zeta=z_probes.copy())

scpyst.track(p_pyst)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure()
plt.subplot(211)
plt.plot(r_probes, p_pyst.px, color='red')
plt.plot(r_probes, p2np(particles.px[:n_probes]), color='blue',
        linestyle='--')
plt.subplot(212)
plt.plot(r_probes, p_pyst.py, color='red')
plt.plot(r_probes, p2np(particles.py[:n_probes]), color='blue',
        linestyle='--')

###########
# Time it #
###########

n_rep = 10
for _ in range(n_rep):
    t1 = time.time()
    spcharge.track(particles)
    t2 = time.time()
    print(f'Time: {(t2-t1)*1e3:.2f} ms')

plt.show()
