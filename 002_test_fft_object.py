import FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FD
import FFT_OpenBoundary_SquareGrid as PIC_FFT
import geom_impact_ellip as ell
from scipy import rand
import numpy as np

R_cham = 1e-1
R_charge = 4e-2
N_part_gen = 100000
Dh = 3e-3

from scipy.constants import e, epsilon_0

qe = e
eps0 = epsilon_0


chamber = ell.ellip_cham_geom_object(x_aper = R_cham, y_aper = R_cham)

picFD = PIC_FD.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh)
picFFT = PIC_FFT.FFT_OpenBoundary_SquareGrid(x_aper = chamber.x_aper, y_aper = chamber.y_aper, Dh = Dh)

# generate particles
x_part = R_charge*(2.*rand(N_part_gen)-1.)
y_part = R_charge*(2.*rand(N_part_gen)-1.)
mask_keep  = x_part**2+y_part**2<R_charge**2
x_part = x_part[mask_keep]
y_part = y_part[mask_keep]

nel_part = 0*x_part+1.

#pic scatter
picFD.scatter(x_part, y_part, nel_part)
picFFT.scatter(x_part, y_part, nel_part)

#pic scatter
picFD.solve()
picFFT.solve()

x_probes = np.linspace(0,R_cham,1000)
y_probes = 0.*x_probes

#pic gather
Ex_FD, Ey_FD = picFD.gather(x_probes, y_probes)
Ex_FFT, Ey_FFT = picFFT.gather(x_probes, y_probes)

E_r_th = map(lambda x: -np.sum(x_part**2+y_part**2<x**2)*qe/eps0/(2*np.pi*x), x_probes)


import pylab as pl
pl.close('all')
pl.plot(x_probes, Ex_FD)
pl.plot(x_probes, Ex_FFT)
pl.plot(x_probes, E_r_th)
#pl.plot(picFFT.xg, picFFT.efx[picFFT.ny/2, :])

pl.show()
