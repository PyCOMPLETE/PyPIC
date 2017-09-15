import sys, os
BIN=os.path.expanduser('../')
sys.path.append(BIN)

from . import FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
from . import FiniteDifferences_Staircase_SquareGrid as PIC_FD
from . import FFT_PEC_Boundary_SquareGrid as PIC_PEC_FFT
from . import geom_impact_ellip as ell
from . import geom_impact_poly as poly
from scipy import rand
import numpy as np



x_aper = 1e-1
y_aper = .5e-1
R_charge = 4e-2
N_part_gen = 100000
Dh = 1e-3

from scipy.constants import e, epsilon_0

qe = e
eps0 = epsilon_0

na = np.array

chamber = poly.polyg_cham_geom_object({'Vx':na([x_aper, -x_aper, -x_aper, x_aper]),
									   'Vy':na([y_aper, y_aper, -y_aper, -y_aper]),
									   'x_sem_ellip_insc':0.99*x_aper,
									   'y_sem_ellip_insc':0.99*y_aper})

picFDSW = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh, sparse_solver = 'PyKLU')
picFD = PIC_FD.FiniteDifferences_Staircase_SquareGrid(chamb = chamber, Dh = Dh, sparse_solver = 'PyKLU')
picFFTPEC = PIC_PEC_FFT.FFT_PEC_Boundary_SquareGrid(x_aper = chamber.x_aper, y_aper = chamber.y_aper, Dh = Dh)

# generate particles
x_part = R_charge*(2.*rand(N_part_gen)-1.)
y_part = R_charge*(2.*rand(N_part_gen)-1.)
mask_keep  = x_part**2+y_part**2<R_charge**2
x_part = x_part[mask_keep]
y_part = y_part[mask_keep]

nel_part = 0*x_part+1.

#pic scatter
picFDSW.scatter(x_part, y_part, nel_part)
picFD.scatter(x_part, y_part, nel_part)
picFFTPEC.scatter(x_part, y_part, nel_part)

#pic scatter
picFDSW.solve()
picFD.solve()
picFFTPEC.solve()


x_probes = np.linspace(0,x_aper,1000)
y_probes = 0.*x_probes

#pic gather
Ex_FDSW, Ey_FDSW = picFDSW.gather(x_probes, y_probes)
Ex_FD, Ey_FD = picFD.gather(x_probes, y_probes)
Ex_FFTPEC, Ey_FFTPEC = picFFTPEC.gather(x_probes, y_probes)



import pylab as pl
pl.close('all')
pl.plot(x_probes, Ex_FDSW, label = 'FD ShorleyWeller')
pl.plot(x_probes, Ex_FD, label = 'FD Staircase')
pl.plot(x_probes, Ex_FFTPEC, label = 'FFT PEC')
#pl.plot(picFFT.xg, picFFT.efx[picFFT.ny/2, :])
pl.legend()
pl.ylabel('Ex on the x axis [V/m]')
pl.xlabel('x [m]')




pl.figure(100)
pl.pcolor(picFDSW.phi.T)
pl.axis('equal')


pl.figure(101)
pl.pcolor(picFFTPEC.phi.T)
pl.axis('equal')


pl.figure(200)
pl.pcolor(picFDSW.efx.T)
pl.axis('equal')



pl.figure(201)
pl.pcolor(picFFTPEC.efx.T)
pl.axis('equal')

pl.figure(300)
pl.pcolor(picFDSW.efy.T)
pl.axis('equal')



pl.figure(301)
pl.pcolor(picFFTPEC.efy.T)
pl.axis('equal')

Ny = picFDSW.Nyg

pl.figure(1003)
pl.plot(picFDSW.phi[:,Ny/2]/picFFTPEC.phi[:,Ny/2])

pl.suptitle('%f'%(np.sum(picFDSW.phi)/np.sum(picFFTPEC.phi)))


pl.show()
