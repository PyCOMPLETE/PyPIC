import PyPIC.FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
import PyPIC.FiniteDifferences_Staircase_SquareGrid as PIC_FD
import PyPIC.FFT_OpenBoundary_SquareGrid as PIC_FFT
import PyPIC.geom_impact_ellip as ell

from scipy import rand
import numpy as np

sigma = .5

R_cham = 10*sigma
Dh = sigma/20.

from scipy.constants import e, epsilon_0

qe = e
eps0 = epsilon_0


chamber = ell.ellip_cham_geom_object(x_aper = R_cham, y_aper = R_cham)

#~ picFD = PIC_FD.FiniteDifferences_Staircase_SquareGrid(chamb = chamber, Dh = Dh)
#~ picFDSW = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh)
picFFT = PIC_FFT.FFT_OpenBoundary_SquareGrid(x_aper = chamber.x_aper, y_aper = chamber.y_aper, Dh = Dh)

YY,XX = np.meshgrid(picFFT.yg, picFFT.xg)
sigmax = sigma
sigmay = sigma
x_beam_pos = 0.
y_beam_pos = 0.
rho_mat=1./(2.*np.pi*sigmax*sigmay)*np.exp(-(XX-x_beam_pos)**2/(2.*sigmax**2)-(YY-y_beam_pos)**2/(2.*sigmay**2))


#pic scatter
#~ picFD.solve(rho = rho_mat)
#~ picFDSW.solve(rho = rho_mat)
picFFT.solve(rho = rho_mat)

x_probes = np.linspace(0,R_cham,1000)
y_probes = 0.*x_probes

#pic gather
#~ Ex_FD, Ey_FD = picFD.gather(x_probes, y_probes)
#~ Ex_FDSW, Ey_FDSW = picFDSW.gather(x_probes, y_probes)
Ex_FFT, Ey_FFT = picFFT.gather(x_probes, y_probes)

E_r_th = [np.sum(rho_mat[:][XX[:]**2+YY[:]**2<x**2])/eps0/(2*np.pi*x)*Dh*Dh for x in x_probes]


import pylab as pl
pl.close('all')
#~ pl.plot(x_probes, Ex_FD, label = 'FD')
#~ pl.plot(x_probes, Ex_FDSW, label = 'FD ShorleyWeller')
pl.plot(x_probes, Ex_FFT, label = 'FFT open')
pl.plot(x_probes, E_r_th, label = 'Analytic')
pl.plot(picFFT.xg, picFFT.efx[:, picFFT.Nyg//2])
pl.legend()
pl.ylabel('Ex on the x axis [V/m]')
pl.xlabel('x [m]')

pl.show()
