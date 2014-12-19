import FiniteDifferences_ShortleyWeller_SquareGrid as PIC
import geom_impact_ellip as ell
from scipy import rand
import numpy as np

R_cham = 1e-1
R_charge = 4e-2
N_part_gen = 100000
Dh = 1e-3

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


import pylab as pl
pl.close('all')
pl.plot(x_probes, Ex)
pl.plot(x_probes, E_r_th)
pl.show()

