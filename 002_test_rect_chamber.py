import FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
import FFT_OpenBoundary_SquareGrid as PIC_FFT
import geom_impact_ellip as ell
import geom_impact_poly as poly
from scipy import rand
import numpy as np

from module_poisson_from_exmple import fft_poisson

x_aper = .5e-1
y_aper = .25e-1
R_charge = 4e-2
N_part_gen = 100000
Dh = 2e-3

from scipy.constants import e, epsilon_0

qe = e
eps0 = epsilon_0

na = np.array

chamber = poly.polyg_cham_geom_object({'Vx':na([x_aper, -x_aper, -x_aper, x_aper]),
									   'Vy':na([y_aper, y_aper, -y_aper, -y_aper]),
									   'x_sem_ellip_insc':0.99*x_aper,
									   'y_sem_ellip_insc':0.99*y_aper})

picFDSW = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh)
picFFT = PIC_FFT.FFT_OpenBoundary_SquareGrid(x_aper = chamber.x_aper, y_aper = chamber.y_aper, Dh = Dh)


# generate particles
x_part = R_charge*(2.*rand(N_part_gen)-1.)
y_part = R_charge*(2.*rand(N_part_gen)-1.)
mask_keep  = x_part**2+y_part**2<R_charge**2
x_part = x_part[mask_keep]
y_part = y_part[mask_keep]

nel_part = 0*x_part+1.

#pic scatter
picFDSW.scatter(x_part, y_part, nel_part)
picFFT.scatter(x_part, y_part, nel_part)

#pic scatter
picFDSW.solve()
picFFT.solve()

x_probes = np.linspace(0,x_aper,1000)
y_probes = 0.*x_probes

#pic gather
Ex_FDSW, Ey_FDSW = picFDSW.gather(x_probes, y_probes)
Ex_FFT, Ey_FFT = picFFT.gather(x_probes, y_probes)

E_r_th = map(lambda x: -np.sum(x_part**2+y_part**2<x**2)*qe/eps0/(2*np.pi*x), x_probes)


import pylab as pl
pl.close('all')
pl.plot(x_probes, Ex_FDSW, label = 'FD ShorleyWeller')
pl.plot(x_probes, Ex_FFT, label = 'FFT open')
pl.plot(x_probes, E_r_th, label = 'Analytic')
#pl.plot(picFFT.xg, picFFT.efx[picFFT.ny/2, :])
pl.legend()
pl.ylabel('Ex on the x axis [V/m]')
pl.xlabel('x [m]')

# I try to use with the dst solver with the correct boundary
xg = picFDSW.xg
yg = picFDSW.yg
x_aper = chamber.x_aper
y_aper = chamber.y_aper

i_min = np.min(np.where(xg>-x_aper)[0])
i_max = np.max(np.where(xg<x_aper)[0])+1
j_min = np.min(np.where(yg>-y_aper)[0])
j_max = np.max(np.where(yg<y_aper)[0])+1


phi = 0*picFDSW.rho
phi[i_min:i_max,j_min:j_max] = fft_poisson(-picFDSW.rho[i_min:i_max,j_min:j_max]/eps0*np.pi**2, Dh)

pl.figure(100)
pl.pcolor(picFDSW.phi.T)
pl.axis('equal')

pl.figure(101)
pl.pcolor(phi.T)
pl.axis('equal')
pl.suptitle('%f'%(np.sum(picFDSW.phi)/np.sum(phi)))

pl.figure(102)
Ny = len(yg)
pl.plot(picFDSW.phi[:,Ny/2]/phi[:,Ny/2])

pl.suptitle('%f'%(np.sum(picFDSW.phi)/np.sum(phi)))


pl.show()
