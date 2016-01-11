import sys, os
BIN=os.path.expanduser('../')
sys.path.append(BIN)
from backwards_compatibility_1_03 import *
import FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
import FiniteDifferences_Staircase_SquareGrid as PIC_FD
import FFT_OpenBoundary_SquareGrid as PIC_FFT
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

picFFT = PIC_FFT.FFT_OpenBoundary_SquareGrid(x_aper = chamber.x_aper, y_aper = chamber.y_aper, Dh = Dh, fftlib = 'pyfftw')

# generate particles
x_part = R_charge*(2.*rand(N_part_gen)-1.)
y_part = R_charge*(2.*rand(N_part_gen)-1.)
mask_keep  = x_part**2+y_part**2<R_charge**2
x_part = x_part[mask_keep]
y_part = y_part[mask_keep]

nel_part = 0*x_part+1.

#pic scatter
picFFT.scatter(x_part, y_part, nel_part)

#pic scatter
picFFT.solve()

x_probes = np.linspace(0,R_cham,1000)
y_probes = 0.*x_probes

#pic gather
Ex_FFT, Ey_FFT = picFFT.gather(x_probes, y_probes)

E_r_th = map(lambda x: -np.sum(x_part**2+y_part**2<x**2)*qe/eps0/(2*np.pi*x), x_probes)


import pylab as pl
pl.close('all')

pl.plot(x_probes, Ex_FFT, label = 'FFT open')
pl.plot(x_probes, E_r_th, label = 'Analytic')
#pl.plot(picFFT.xg, picFFT.efx[picFFT.ny/2, :])
pl.legend()
pl.ylabel('Ex on the x axis [V/m]')
pl.xlabel('x [m]')

self = picFFT
tmprho = 0.*self.fgreen
tmprho[:self.ny, :self.nx] = self.rho.T

pl.figure(102)
pl.semilogy(np.abs(self.fft2(tmprho).flatten()))
pl.semilogy(np.abs(np.fft.fft2(tmprho).flatten()))


pl.figure(103)
pl.semilogy(np.abs(self.fgreentr.flatten()))

pl.figure(104)
pl.semilogy(np.abs(self.fgreen.flatten()))


fftphi = self.fft2(tmprho) * self.fgreentr

pl.figure(100)
pl.semilogy(np.abs(fftphi.flatten()))

print 'type(fftphi[0,0])', type(fftphi[0,0])
print 'type(self.fgreentr[0,0])',type(self.fgreentr[0,0])
tmpphi = np.fft.ifft2(fftphi)
pl.figure(101)
pl.semilogy(np.abs(tmpphi.flatten()))


self.phi = 1./(4. * np.pi * eps0)*np.real(tmpphi[:self.ny, :self.nx]).T

self.efx[1:self.Nxg-1,:] = self.phi[0:self.Nxg-2,:] - self.phi[2:self.Nxg,:];  #central difference on internal nodes
self.efy[:,1:self.Nyg-1] = self.phi[:,0:self.Nyg-2] - self.phi[:,2:self.Nyg];  #central difference on internal nodes


self.efy = self.efy/(2*self.Dh)
self.efx = self.efx/(2*self.Dh)


pl.show()




