import sys
sys.path.append('..')

import PyPIC.FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
import PyPIC.FiniteDifferences_Staircase_SquareGrid as PIC_FD
import PyPIC.FFT_OpenBoundary as PIC_FFT
try:
    from CyFPPS import PyFPPS as PIC_FPPS
except ImportError:
    print("Not possible to import PyFPPS, replaced with FFT_Open")
    PIC_FPPS = None
from PyPIC.MultiGrid import AddInternalGrid

import PyPIC.geom_impact_ellip as ell
from scipy import rand
import numpy as np

R_cham = 1e-1
R_charge = 4e-2
N_part_gen = 1000000
Dh = 1e-3

# Settings for dual grid
x_min_internal = -R_charge*1.1
x_max_internal = R_charge*1.1
y_min_internal = -R_charge*1.2
y_max_internal = R_charge*1.2
Dh_main = 1e-3
Dh_internal = .2e-3
N_nodes_discard = 3


from scipy.constants import e, epsilon_0

qe = e
eps0 = epsilon_0


chamber = ell.ellip_cham_geom_object(x_aper = R_cham, y_aper = R_cham)

picFD = PIC_FD.FiniteDifferences_Staircase_SquareGrid(chamb = chamber, Dh = Dh)
picFDSW = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh)
picFFT = PIC_FFT.FFT_OpenBoundary(x_aper = chamber.x_aper, y_aper = chamber.y_aper, dx = Dh/2., dy = Dh, fftlib='pyfftw')
picFFTSq = PIC_FFT.FFT_OpenBoundary(x_aper = chamber.x_aper, y_aper = chamber.y_aper, Dh = Dh, fftlib='pyfftw')
if PIC_FPPS: picFPPS = PIC_FPPS(200,200,a=R_cham,solverType='Uniform')
# build dual grid
pic_main = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh_main)
pic_dualgrid = AddInternalGrid(pic_main, x_min_internal, x_max_internal, y_min_internal, 
                                y_max_internal, Dh_internal, N_nodes_discard)

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
picFFTSq.scatter(x_part, y_part, nel_part)
if PIC_FPPS: picFPPS.scatter(x_part,y_part,nel_part)
pic_dualgrid.scatter(x_part, y_part, nel_part)

#pic scatter
picFD.solve()
picFDSW.solve()
picFFT.solve()
picFFTSq.solve()
if PIC_FPPS:picFPPS.solve()
pic_dualgrid.solve()

x_probes = np.linspace(0,R_cham,1000)
y_probes = 0.*x_probes

#pic gather
Ex_FD, Ey_FD = picFD.gather(x_probes, y_probes)
Ex_FDSW, Ey_FDSW = picFDSW.gather(x_probes, y_probes)
Ex_FFT, Ey_FFT = picFFT.gather(x_probes, y_probes)
Ex_FFTSq, Ey_FFTSq = picFFTSq.gather(x_probes, y_probes)
if PIC_FPPS: Ex_FPPS,Ey_FPPS = picFPPS.gather(x_probes,y_probes)
Ex_dualgrid, Ey_dualgrid = pic_dualgrid.gather(x_probes, y_probes)
E_r_th = [-np.sum(x_part**2+y_part**2<x**2)*qe/eps0/(2*np.pi*x) for x in x_probes]


import pylab as pl
pl.close('all')

pl.plot(x_probes, Ex_FD, label = 'FD')
pl.plot(x_probes, Ex_FDSW, label = 'FD ShortleyWeller')
pl.plot(x_probes, Ex_FFT, label = 'FFT open rect.')
pl.plot(x_probes, Ex_FFTSq, label = 'FFT open square')
if PIC_FPPS: pl.plot(x_probes, Ex_FPPS, label = 'FPPS')
pl.plot(x_probes, Ex_dualgrid, label = 'Dual grid')
pl.plot(x_probes, E_r_th, label = 'Analytic')
pl.legend()
pl.ylabel('Ex on the x axis [V/m]')
pl.xlabel('x [m]')


pl.show()
