import pylab as pl
import numpy as np
from scipy import rand
import geom_impact_poly as poly
import FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
import FFT_OpenBoundary_SquareGrid as PIC_FFT
import FFT_PEC_Boundary_SquareGrid as PIC_PEC_FFT

na = np.array

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

x_aper = 7.
y_aper = 7.

Dh = x_aper/128*2.

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
									   

picFFT = PIC_FFT.FFT_OpenBoundary_SquareGrid(x_aper = chamber.x_aper, y_aper = chamber.y_aper, Dh = Dh, fftlib='numpy')


picFFT.scatter(x_part, y_part, nel_part)

N_rep = 1000

picFFT.solve()

self = picFFT

rho = self.rho
from scipy.constants import epsilon_0 as eps0


tmprho = 0.*self.fgreen
tmprho[:self.ny, :self.nx] = rho.T


fftphi = self.fft2(tmprho) * self.fgreentr
tmpphi = self.ifft2(fftphi)
phi = 1./(4. * np.pi * eps0)*np.real(tmpphi[:self.ny, :self.nx]).T


import pyfftw
fft_first = pyfftw.builders.fft(tmprho[:self.ny, :], axis = 1)
transf1 = 0.*self.fgreentr
transf1[:self.ny, :] = fft_first(tmprho[:self.ny, :])
fft_second = pyfftw.builders.fft(transf1, axis = 0)
fftphi_new = fft_second(transf1)* self.fgreentr


ifft_first = pyfftw.builders.ifft(fftphi_new, axis = 0)
itransf1 = ifft_first(fftphi_new)
ifft_second = pyfftw.builders.ifft(itransf1[:self.ny, :], axis = 1)
tmpphi_new = ifft_second(itransf1[:self.ny, :])

def fft2(x):
	tmp = (x*(1.+1j))*0.
	tmp[:self.ny, :] = fft_first(x[:self.ny, :])
	return fft_second(tmp)
	
def ifft2(x):
	tmp = ifft_first(x)
	res = 0*x
	res[:self.ny, :] = np.real(ifft_second(tmp[:self.ny, :]))
	return res
	
fftphi = self.fft2(tmprho) * self.fgreentr
tmpphi = self.ifft2(fftphi)
phi = 1./(4. * np.pi * eps0)*np.real(tmpphi[:self.ny, :self.nx]).T
	
	
fftphi = fft2(tmprho) * self.fgreentr
tmpphi = ifft2(fftphi)
phinew = 1./(4. * np.pi * eps0)*np.real(tmpphi[:self.ny, :self.nx]).T

N_rep = 100

import time
t_start_fft2 = time.mktime(time.localtime())
for _ in xrange(N_rep):
	fftphi = self.fft2(tmprho) * self.fgreentr
	tmpphi = self.ifft2(fftphi)
	phi = 1./(4. * np.pi * eps0)*np.real(tmpphi[:self.ny, :self.nx]).T
t_stop_fft2 = time.mktime(time.localtime())
t_fft2 = t_stop_fft2-t_start_fft2
print 't_fft2', t_fft2


t_start_reduc = time.mktime(time.localtime())
for _ in xrange(N_rep):
	fftphi = fft2(tmprho) * self.fgreentr
	tmpphi = ifft2(fftphi)
	phinew = 1./(4. * np.pi * eps0)*np.real(tmpphi[:self.ny, :self.nx]).T
t_stop_reduc = time.mktime(time.localtime())
t_reduc = t_stop_reduc-t_start_reduc
print 't_reduc', t_reduc



