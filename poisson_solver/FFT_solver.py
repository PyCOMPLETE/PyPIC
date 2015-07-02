'''
FFT Poisson solvers for PyPIC
@author Stefan Hegglin, Adrian Oeftiger, Giovanni Iadarola
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
        return phi

