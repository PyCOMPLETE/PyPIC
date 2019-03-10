import pylab as pl
import numpy as np
from scipy import rand
import time
import geom_impact_poly as poly
import FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
import FFT_OpenBoundary_SquareGrid as PIC_FFT
import FFT_PEC_Boundary_SquareGrid as PIC_PEC_FFT

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
                                       


picFFT = PIC_FFT.FFT_OpenBoundary_SquareGrid(x_aper = chamber.x_aper, y_aper = chamber.y_aper, Dh = Dh)


picFFT.scatter(x_part, y_part, nel_part)

data = picFFT.fgreen

N_rep = 1000


t_start_npfft = time.mktime(time.localtime())
for _ in xrange(N_rep):
    transf = np.fft.fft2(data)
    itransf = np.real(np.fft.ifft2(transf*data))
    
t_stop_npfft = time.mktime(time.localtime())
t_npfft = t_stop_npfft-t_start_npfft
print 't_npfft', t_npfft



import pyfftw

fftobj = pyfftw.builders.fft2(data.copy())
temptransf = fftobj(data)
ifftobj = pyfftw.builders.ifft2(temptransf)

t_start_npfftw = time.mktime(time.localtime())
for _ in xrange(N_rep):
    transfw = fftobj(data)
    itransfw = ifftobj(transfw)
t_stop_npfftw = time.mktime(time.localtime())
t_npfftw = t_stop_npfftw-t_start_npfftw
print 't_npfftw', t_npfftw


