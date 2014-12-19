import FiniteDifferences_ShortleyWeller_SquareGrid as PIC
import geom_impact_ellip as ell
from scipy import rand
import numpy as np

R_cham = 1e-1
R_charge = 4e-2
N_part_gen = 100000
Dh = 3e-3

eps0=8.8541878176e-12;
qe=1.602176565e-19;


chamber = ell.ellip_cham_geom_object(x_aper = R_cham, y_aper = R_cham)

pic = PIC.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh)

# generate particles
x_part = R_charge*(2.*rand(N_part_gen)-1.)
y_part = R_charge*(2.*rand(N_part_gen)-1.)
mask_keep  = x_part**2+y_part**2<R_charge**2
x_part = x_part[mask_keep]
y_part = y_part[mask_keep]

nel_part = 0*x_part+1.

#pic scatter
pic.scatter(x_part, y_part, nel_part)

pic.solve()

x_probes = np.linspace(0,R_cham,1000)
y_probes = 0.*x_probes


E_r_th = map(lambda x: -np.sum(x_part**2+y_part**2<x**2)*qe/eps0/(2*np.pi*x), x_probes)
#pic gather
Ex, Ey = pic.gather(x_probes, y_probes)

dx = pic.Dh
dy = pic.Dh
nx = len(pic.xg)
ny = len(pic.yg)


mx = -dx / 2 + np.arange(nx + 1) * dx
my = -dy / 2 + np.arange(ny + 1) * dy
x, y = np.meshgrid(mx, my)
r2 = x ** 2 + y ** 2
# Antiderivative
tmpfgreen = -1 / 2 * (-3 * x * y + x * y * np.log(r2)
		   + x * x * np.arctan(y / x) + y * y * np.arctan(x / y)) # * 2 / dx / dy
		   
fgreen = np.zeros((2 * ny, 2 * nx))
# Integration and circular Green's function
fgreen[:ny, :nx] = tmpfgreen[1:, 1:] + tmpfgreen[:-1, :-1] - tmpfgreen[1:, :-1] - tmpfgreen[:-1, 1:]
fgreen[ny:, :nx] = fgreen[ny:0:-1, :nx]
fgreen[:ny, nx:] = fgreen[:ny, nx:0:-1]
fgreen[ny:, nx:] = fgreen[ny:0:-1, nx:0:-1]

tmprho = 0.*fgreen
tmprho[:ny, :nx] = pic.rho.T

fftphi = np.fft.fft2(tmprho) * np.fft.fft2(fgreen)

tmpphi = np.fft.ifft2(fftphi)
phi = 1./(4. * np.pi * eps0)*np.abs(tmpphi[:ny, :nx])

ey, ex = np.gradient(phi, dy, dx)


import pylab as pl
pl.close('all')
pl.plot(x_probes, Ex)
pl.plot(x_probes, E_r_th)
pl.plot(pic.xg, ex[ny/2, :])

pl.show()
