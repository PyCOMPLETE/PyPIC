import PyPIC.FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
import PyPIC.FFT_OpenBoundary_SquareGrid as PIC_FFT
import PyPIC.geom_impact_ellip as ell

from scipy import rand
import numpy as np

R_cham = 1e-1
R_charge = 4e-2
N_part_gen = 100000
Dh = 2e-3

from scipy.constants import e, epsilon_0

qe = e
eps0 = epsilon_0


chamber = ell.ellip_cham_geom_object(x_aper = R_cham, y_aper = R_cham)

picFDSW = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh)
#~ picFDSW = PIC_FDSW.FiniteDifferences_Staircase_SquareGrid(chamb = chamber, Dh = Dh)


# generate particles
x_part = R_charge*(2.*rand(N_part_gen)-1.)
y_part = R_charge*(2.*rand(N_part_gen)-1.)
mask_keep  = x_part**2+y_part**2<R_charge**2
x_part = x_part[mask_keep]
y_part = y_part[mask_keep]

nel_part = 0*x_part+1.

#pic scatter
picFDSW.scatter(x_part, y_part, nel_part)

#pic scatter
picFDSW.solve()

x_probes = np.linspace(0,R_cham,1000)
y_probes = 0.*x_probes

#pic gather
Ex_FDSW, Ey_FDSW = picFDSW.gather(x_probes, y_probes)

E_r_th = [-np.sum(x_part**2+y_part**2<x**2)*qe/eps0/(2*np.pi*x) for x in x_probes]


import pylab as pl
pl.close('all')
pl.figure(1)
pl.plot(x_probes, Ex_FDSW, label = 'FD ShorleyWeller')
pl.plot(x_probes, E_r_th, label = 'Analytic')
#pl.plot(picFFT.xg, picFFT.efx[picFFT.ny/2, :])
pl.legend()
pl.ylabel('Ex on the x axis [V/m]')
pl.xlabel('x [m]')

pl.figure(2)
pl.pcolormesh(picFDSW.xg, picFDSW.yg, picFDSW.efx.T)
pl.axis('equal')
theta = np.linspace(0, 2*np.pi, 1000)
pl.plot(R_cham*np.cos(theta), R_cham*np.sin(theta), color='k')


pl.show()
