#----------------------------------------------------------------------
#
#                           CERN
#
#     European Organization for Nuclear Research
#
#
#     This file is part of the code:
#
#
#                   PyPIC Version 2.0.0
#
#
#     Author and contact:   Giovanni IADAROLA
#                           BE-ABP Group
#                           CERN
#                           CH-1211 GENEVA 23
#                           SWITZERLAND
#                           giovanni.iadarola@cern.ch
#
#                contact:   Giovanni RUMOLO
#                           BE-ABP Group
#                           CERN
#                           CH-1211 GENEVA 23
#                           SWITZERLAND
#                           giovanni.rumolo@cern.ch
#
#
#
#     Copyright  CERN,  Geneva  2011  -  Copyright  and  any   other
#     appropriate  legal  protection  of  this  computer program and
#     associated documentation reserved  in  all  countries  of  the
#     world.
#
#     Organizations collaborating with CERN may receive this program
#     and documentation freely and without charge.
#
#     CERN undertakes no obligation  for  the  maintenance  of  this
#     program,  nor responsibility for its correctness,  and accepts
#     no liability whatsoever resulting from its use.
#
#     Program  and documentation are provided solely for the use  of
#     the organization to which they are distributed.
#
#     This program  may  not  be  copied  or  otherwise  distributed
#     without  permission. This message must be retained on this and
#     any other authorized copies.
#
#     The material cannot be sold. CERN should be  given  credit  in
#     all references.
#----------------------------------------------------------------------

import numpy as np
import scipy.sparse as scsp
from scipy.sparse.linalg import spsolve
import scipy.sparse.linalg as ssl
from vectsum import vectsum
from PyPIC_Scatter_Gather import PyPIC_Scatter_Gather
from scipy.constants import e, epsilon_0

na = lambda x:np.array([x])

qe=e
eps0=epsilon_0

class FFT_OpenBoundary(PyPIC_Scatter_Gather):
    #@profile
    def __init__(self, x_aper, y_aper, Dh=None, dx=None, dy=None,
                 fftlib='pyfftw', verbose=True):

        self.verbose = verbose
        if verbose:
            print 'Start PIC init.:'
            print 'FFT, Open Boundary'


        if dx!=None and dy!=None:
            assert(Dh==None)

        elif Dh!=None:
            assert(dx==None and dy==None)
            dx = Dh
            dy = Dh

        else:
            raise ValueError('Dh or dx and dy must be specified!!!')

        super(FFT_OpenBoundary, self).__init__(x_aper, y_aper, dx, dy,
                                               verbose=verbose)


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
                if self.verbose:
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
                if self.verbose:
                    print 'Failed to import pyfftw'
                    print 'Got exception: ', err
                    print 'Using numpy fft'
                self.fft2 = np.fft.fft2
                self.ifft2 = np.fft.ifft2
        elif fftlib == 'numpy':
            if self.verbose:
                print 'Using numpy FFT'
            self.fft2 = np.fft.fft2
            self.ifft2 = np.fft.ifft2
        else:
            raise ValueError('fftlib not recognized!!!!')

        self.fgreen = fgreen
        self.fgreentr = np.fft.fft2(fgreen).copy()
        self.rho = np.zeros((self.Nxg,self.Nyg))
        self.phi = np.zeros((self.Nxg,self.Nyg))
        self.efx = np.zeros((self.Nxg,self.Nyg))
        self.efy = np.zeros((self.Nxg,self.Nyg))
        self.Dh = Dh
        self.nx = nx
        self.ny = ny


    #@profile
    def solve(self, rho = None, flag_verbose = False):
        if rho == None:
            rho = self.rho

        tmprho = 0.*self.fgreen
        tmprho[:self.ny, :self.nx] = rho.T

        fftphi = self.fft2(tmprho) * self.fgreentr

        tmpphi = self.ifft2(fftphi)
        self.phi = 1./(4. * np.pi * eps0)*np.real(tmpphi[:self.ny, :self.nx]).T

        self.efx[1:self.Nxg-1,:] = self.phi[0:self.Nxg-2,:] - self.phi[2:self.Nxg,:];  #central difference on internal nodes
        self.efy[:,1:self.Nyg-1] = self.phi[:,0:self.Nyg-2] - self.phi[:,2:self.Nyg];  #central difference on internal nodes


        self.efy = self.efy/(2*self.dy)
        self.efx = self.efx/(2*self.dx)





