'''
FFT Poisson solvers for PyPIC
@author Stefan Hegglin, Adrian Oeftiger, Giovanni Iadarola
Implementation/Logic: Giovanni Idadarola
New interface: Stefan Hegglin, Adrian Oeftiger
'''

from __future__ import division

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl
from scipy.constants import epsilon_0

from poisson_solver import PoissonSolver
from FD_solver import compute_new_mesh_properties


class FFT_OpenBoundary_SquareGrid(PoissonSolver):
    '''
    Wrapper for the old PyPIC FFT open boundary solver
    '''
    def __init__(self, x_aper, y_aper, Dh, fftlib='pyfftw'):
        na = lambda x:np.array([x])
        params = compute_new_mesh_properties(
                     x_aper, y_aper, Dh, ext_boundary=True) #always True!

        self.Dh, self.xg, self.Nxg, self.bias_x, self.yg, self.Nyg, self.bias_y = params
        dx = self.xg[1] - self.xg[0]
        dy = self.yg[1] - self.yg[0]

        nx = len(self.xg)
        ny = len(self.yg)
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

        if fftlib == 'pyfftw':
            try:
                import pyfftw
                print 'Using PyFFTW'
                #prepare fftw's
                tmprho = fgreen.copy()
                fft_first = pyfftw.builders.fft(tmprho[:ny, :].copy(), axis = 1)
                transf1 = (fgreen*(1.+1j))*0.
                transf1[:ny, :] = fft_first(tmprho[:ny, :].copy())
                fft_second = pyfftw.builders.fft(transf1.copy(), axis = 0)
                fftphi_new = fft_second(transf1.copy())* fgreen
                ifft_first = pyfftw.builders.ifft(fftphi_new.copy(), axis = 0)
                itransf1 = ifft_first(fftphi_new.copy())
                ifft_second = pyfftw.builders.ifft(itransf1[:ny, :].copy(), axis = 1)

                def fft2(x):
                    tmp = (x*(1.+1j))*0.
                    tmp[:ny, :] = fft_first(x[:ny, :])
                    return fft_second(tmp)

                def ifft2(x):
                    tmp = ifft_first(x)
                    res = 0*x
                    res[:ny, :] = np.real(ifft_second(tmp[:ny, :]))
                    return res

                self.fft2 = fft2
                self.ifft2 = ifft2

            except ImportError as err:
                print 'Failed to import pyfftw'
                print 'Got exception: ', err
                print 'Using numpy fft'
                self.fft2 = np.fft.fft2
                self.ifft2 = np.fft.ifft2
        elif fftlib == 'numpy':
                print 'Using numpy FFT'
                self.fft2 = np.fft.fft2
                self.ifft2 = np.fft.ifft2
        else:
            raise ValueError('fftlib not recognized!')
        self.fgreen = fgreen
        self.fgreentr = np.fft.fft2(fgreen).copy()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy

    def poisson_solve(self, mesh_charges):
        tmprho = 0.*self.fgreen
        mesh_charges = mesh_charges.reshape(self.Nyg, self.Nxg) / (self.dx*self.dy)
        tmprho[:self.ny, :self.nx] = mesh_charges

        fftphi = self.fft2(tmprho) * self.fgreentr

        tmpphi = self.ifft2(fftphi)
        phi = 1./(4. * np.pi * epsilon_0)*np.real(tmpphi[:self.ny, :self.nx]).T
        phi = phi.reshape(self.Nxg, self.Nyg).T.flatten()
        return phi*2 #magic number... TODO find out why this is needed!!


class FFT_PEC_Boundary_SquareGrid(PoissonSolver):
    '''
    Wrapper for the old PyPIC FFT perdiodic boundary solver
    '''
    def __init__(self, x_aper, y_aper, Dh, fftlib='pyfftw'):
        na = lambda x:np.array([x])
        params = compute_new_mesh_properties(
                     x_aper, y_aper, Dh, ext_boundary=True) #always True!

        self.Dh, self.xg, self.Nxg, self.bias_x, self.yg, self.Nyg, self.bias_y = params
        self.i_min = np.min(np.where(self.xg>=-x_aper)[0])
        self.i_max = np.max(np.where(self.xg<=x_aper)[0])+1
        self.j_min = np.min(np.where(self.yg>=-y_aper)[0])
        self.j_max = np.max(np.where(self.yg<=y_aper)[0])+1

        dummy = np.zeros((self.Nxg,self.Nyg))


        m, n = dummy[self.i_min:self.i_max,self.j_min:self.j_max].shape;

        xx = np.arange(1,m+0.5,1);
        yy = np.arange(1,n+0.5,1);

        YY, XX = np.meshgrid(yy,xx)
        self.green = 4.*epsilon_0*(np.sin(XX/2*np.pi/float(m+1.))**2/self.Dh**2+\
                     np.sin(YY/2.*np.pi/float(n+1.))**2/self.Dh**2);

        # handle border
        [xn, yn]=np.meshgrid(self.xg,self.yg)

        xn=xn.T
        xn=xn.flatten()

        yn=yn.T
        yn=yn.flatten()
        #% xn and yn are stored such that the external index is on x 

        flag_outside_n=np.logical_or(np.abs(xn)>x_aper,np.abs(yn)>y_aper)
        flag_inside_n=~(flag_outside_n)


        flag_outside_n_mat=np.reshape(flag_outside_n,(self.Nyg,self.Nxg),'F');
        flag_outside_n_mat=flag_outside_n_mat.T
        [gx,gy]=np.gradient(np.double(flag_outside_n_mat));
        gradmod=abs(gx)+abs(gy);
        flag_border_mat=np.logical_and((gradmod>0), flag_outside_n_mat);
        self.flag_border_mat = flag_border_mat

        if fftlib == 'pyfftw':
            try:
                import pyfftw
                rhocut = dummy[self.i_min:self.i_max,self.j_min:self.j_max]
                m, n = rhocut.shape;
                tmp = np.zeros((2*m + 2, n))
                self.ffti = pyfftw.builders.fft(tmp.copy(), axis=0)
                tmp = np.zeros((m, 2*n + 2))
                self.fftj = pyfftw.builders.fft(tmp.copy(), axis=1)
            except ImportError as err:
                print 'Failed to import pyfftw'
                print 'Got exception: ', err
                print 'Using numpy fft'
                self.ffti = lambda xx: np.fft.fft(xx, axis=0)
                self.fftj = lambda xx: np.fft.fft(xx, axis=1)
        elif fftlib == 'numpy':
            self.ffti = lambda xx: np.fft.fft(xx, axis=0)
            self.fftj = lambda xx: np.fft.fft(xx, axis=1)
        else:
            raise ValueError('fftlib not recognized!!!!')

    def dst2(self, x):
        m, n = x.shape;

        #transform along i
        tmp = np.zeros((2*m + 2, n))
        tmp[1:m+1, :] = x
        tmp=-(self.ffti(tmp).imag)
        xtr_i = np.sqrt(2./(m+1.))*tmp[1:m+1, :]

        #transform along j
        tmp = np.zeros((m, 2*n + 2))
        tmp[:, 1:n+1] = xtr_i
        tmp=-(self.fftj(tmp).imag)
        x_bar = np.sqrt(2./(n+1.))*tmp[:, 1:n+1]

        return x_bar

    def poisson_solve(self, mesh_charges):
        mesh_charges = mesh_charges.reshape(self.Nyg, self.Nxg) / (self.Dh*self.Dh)
        #rhocut = mesh_charges[self.i_min:self.i_max,self.j_min:self.j_max]
        rhocut = mesh_charges[self.j_min:self.j_max, self.i_min:self.i_max]
        rho_bar =  self.dst2(rhocut)
        phi_bar = rho_bar.T/self.green
        phi = np.zeros((self.Nxg, self.Nyg))
        phi[self.i_min:self.i_max,self.j_min:self.j_max] = self.dst2(phi_bar).copy()
        phi = phi.reshape(self.Nxg, self.Nyg).T.flatten()
        return phi
