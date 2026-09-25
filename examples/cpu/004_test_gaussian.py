import PyPIC.FFT_OpenBoundary_SquareGrid as PIC_FFT
import PyPIC.geom_impact_ellip as ell

import numpy as np

sigma = .5

R_cham = 10*sigma
Dh = sigma/20.

from scipy.constants import e, epsilon_0

qe = e
eps0 = epsilon_0


chamber = ell.ellip_cham_geom_object(x_aper = R_cham, y_aper = R_cham)

picFFT = PIC_FFT.FFT_OpenBoundary_SquareGrid(x_aper = chamber.x_aper, y_aper = chamber.y_aper, Dh = Dh, fftlib='numpy')

YY,XX = np.meshgrid(picFFT.yg, picFFT.xg)
sigmax = sigma
sigmay = sigma
x_beam_pos = 0.
y_beam_pos = 0.
rho_mat=1./(2.*np.pi*sigmax*sigmay)*np.exp(-(XX-x_beam_pos)**2/(2.*sigmax**2)-(YY-y_beam_pos)**2/(2.*sigmay**2))


#pic scatter
picFFT.solve(rho = rho_mat)

x_probes = np.linspace(0,R_cham,1000)
y_probes = 0.*x_probes

#pic gather
Ex_FFT, Ey_FFT = picFFT.gather(x_probes, y_probes)

# Analytic field of a round Gaussian with unit line charge, including r=0.
E_r_th = np.divide(-np.expm1(-x_probes**2 / (2 * sigma**2)),
                   2 * np.pi * eps0 * x_probes,
                   out=np.zeros_like(x_probes), where=x_probes != 0)


import pylab as pl
pl.close('all')
pl.plot(x_probes, Ex_FFT, label = 'FFT open')
pl.plot(x_probes, E_r_th, label = 'Analytic')
pl.plot(picFFT.xg, picFFT.efx[:, picFFT.Nyg//2])
pl.legend()
pl.ylabel('Ex on the x axis [V/m]')
pl.xlabel('x [m]')

pl.show()
