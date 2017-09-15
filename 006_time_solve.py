import pylab as pl
import numpy as np
from scipy import rand
from . import geom_impact_poly as poly
from . import FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
from . import FFT_OpenBoundary_SquareGrid as PIC_FFT
from . import FFT_PEC_Boundary_SquareGrid as PIC_PEC_FFT

na = np.array
Dh =1e-1
N_part_gen = 100000

tree = [[0,0],
		[1.,0],
		[1., 1,],
		[5.,1.],
		[2.,4.],
		[4,4],
		[2,7],
		[3,7],
		[1,9],
		[2,9],
		[0,11]]
		
tree=np.array(tree)
x_tree = tree[:,0]
y_tree = tree[:,1]

y_tree -= 6.

x_aper = 6.
y_aper = 7.

x_tree = np.array([0.]+ list(x_tree)+[0.])
y_tree = np.array([-y_aper]+ list(y_tree)+[y_aper])


		


x_part = x_aper*(2.*rand(N_part_gen)-1.)
y_part = y_aper*(2.*rand(N_part_gen)-1.)

x_on_tree = np.interp(y_part, y_tree, x_tree)

mask_keep = np.logical_and(np.abs(x_part)<x_on_tree, np.abs(x_part)>x_on_tree*0.8)
x_part = x_part[mask_keep]
y_part = y_part[mask_keep]

nel_part = 0*x_part+1.


		


chamber = poly.polyg_cham_geom_object({'Vx':na([x_aper, -x_aper, -x_aper, x_aper]),
									   'Vy':na([y_aper, y_aper, -y_aper, -y_aper]),
									   'x_sem_ellip_insc':0.99*x_aper,
									   'y_sem_ellip_insc':0.99*y_aper})
									   
picFDSW = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh)
picFFTPEC = PIC_PEC_FFT.FFT_PEC_Boundary_SquareGrid(x_aper = chamber.x_aper, y_aper = chamber.y_aper, Dh = Dh, fftlib='pyfftw')
picFFT = PIC_FFT.FFT_OpenBoundary_SquareGrid(x_aper = chamber.x_aper, y_aper = chamber.y_aper, Dh = Dh, fftlib='pyfftw')

picFDSW.scatter(x_part, y_part, nel_part)
picFFTPEC.scatter(x_part, y_part, nel_part)
picFFT.scatter(x_part, y_part, nel_part)

N_rep = 1000

import time
t_start_sw = time.mktime(time.localtime())
for _ in range(N_rep):
	picFDSW.solve()
t_stop_sw = time.mktime(time.localtime())
t_sw = t_stop_sw-t_start_sw
print('t_sw', t_sw)


t_start_fftpec = time.mktime(time.localtime())
for _ in range(N_rep):
	picFFTPEC.solve()
t_stop_fftpec = time.mktime(time.localtime())
t_fftpec = t_stop_fftpec-t_start_fftpec
print('t_fftpec', t_fftpec)


t_start_fftopen = time.mktime(time.localtime())
for _ in range(N_rep):
	picFFT.solve()
t_stop_fftopen = time.mktime(time.localtime())
t_fftopen = t_stop_fftopen-t_start_fftopen
print('t_fftopen', t_fftopen)


