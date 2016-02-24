import FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
import FiniteDifferences_Staircase_SquareGrid as PIC_FD
import FFT_OpenBoundary_SquareGrid as PIC_FFT
from CyFPPS import PyFPPS as PIC_FPPS
import geom_impact_ellip as ell
from scipy import rand
import numpy as np

R_cham = 1e-1
R_charge = 4e-2
N_part_gen = 100000
Dh = 1e-3

from scipy.constants import e, epsilon_0

qe = e
eps0 = epsilon_0


chamber = ell.ellip_cham_geom_object(x_aper = R_cham, y_aper = R_cham)

picFD = PIC_FD.FiniteDifferences_Staircase_SquareGrid(chamb = chamber, Dh = Dh)
picFDSW = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh)
picFFT = PIC_FFT.FFT_OpenBoundary_SquareGrid(x_aper = chamber.x_aper, y_aper = chamber.y_aper, Dh = Dh, fftlib='pyfftw')
picFPPS = PIC_FPPS(40,40,a=Dh,solverType='Uniform')

# generate particles
x_part = R_charge*(2.*rand(N_part_gen)-1.)
y_part = R_charge*(2.*rand(N_part_gen)-1.)
mask_keep  = x_part**2+y_part**2<R_charge**2
x_part = x_part[mask_keep]
y_part = y_part[mask_keep]

nel_part = 0*x_part+1.

#pic scatter
picFD.scatter(x_part, y_part, nel_part)
picFDSW.scatter(x_part, y_part, nel_part)
picFFT.scatter(x_part, y_part, nel_part)
picFPPS.scatter(x_part,y_part,nel_part)

#pic scatter
picFD.solve()
picFDSW.solve()
picFFT.solve()
picFPPS.solve()

x_probes = np.linspace(0,R_cham,1000)
y_probes = 0.*x_probes

#pic gather
Ex_FD, Ey_FD = picFD.gather(x_probes, y_probes)
Ex_FDSW, Ey_FDSW = picFDSW.gather(x_probes, y_probes)
Ex_FFT, Ey_FFT = picFFT.gather(x_probes, y_probes)
Ex_FPPS,Ey_FPPS = picFPPS.gather(x_probes,y_probes)

E_r_th = map(lambda x: -np.sum(x_part**2+y_part**2<x**2)*qe/eps0/(2*np.pi*x), x_probes)


import pylab as pl
pl.close('all')
pl.plot(x_probes, Ex_FD, label = 'FD')
pl.plot(x_probes, Ex_FDSW, label = 'FD ShorleyWeller')
pl.plot(x_probes, Ex_FFT, label = 'FFT open')
pl.plot(x_probes, Ex_FPPS, label = 'FPPS')
pl.plot(x_probes, E_r_th, label = 'Analytic')
#pl.plot(picFFT.xg, picFFT.efx[picFFT.ny/2, :])
pl.legend()
pl.ylabel('Ex on the x axis [V/m]')
pl.xlabel('x [m]')

pl.show()
