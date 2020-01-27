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
#                  PyPIC Version 2.4.5                     
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
from .PyPIC_Scatter_Gather import PyPIC_Scatter_Gather
from scipy.constants import e, epsilon_0

na = lambda x:np.array([x])

qe=e
eps0=epsilon_0

class FFT_OpenBoundary(PyPIC_Scatter_Gather):
    #@profile
    def __init__(self, x_aper, y_aper, Dh=None, dx=None, dy=None, fftlib = 'pyfftw'):
        
        print('Start PIC init.:')
        print('FFT, Open Boundary')


        if dx is not None and dy is not None:
            assert(Dh is None)

        elif Dh!=None:
            assert(dx is None and dy is None)
            dx = Dh
            dy = Dh

        else:
            raise ValueError('Dh or dx and dy must be specified!!!')
        
        super(FFT_OpenBoundary, self).__init__(x_aper, y_aper, dx, dy)


        nx = len(self.xg)
        ny = len(self.yg)


        mx = -dx / 2 + np.arange(nx + 1) * dx
        my = -dy / 2 + np.arange(ny + 1) * dy
        x, y = np.meshgrid(mx, my)
        r2 = x ** 2 + y ** 2
        # Antiderivative
        tmpfgreen = -(-3 * x * y + x * y * np.log(r2)
                    + x * x * np.arctan(y / x) + y * y * np.arctan(x / y)) # * 2 / dx / dy
                   
        fgreen = np.zeros((2 * ny, 2 * nx))
        # Integration and circular Green's function
        fgreen[:ny, :nx] = tmpfgreen[1:, 1:] + tmpfgreen[:-1, :-1] - tmpfgreen[1:, :-1] - tmpfgreen[:-1, 1:]
        fgreen[ny:, :nx] = fgreen[ny:0:-1, :nx]
        fgreen[:ny, nx:] = fgreen[:ny, nx:0:-1]
        fgreen[ny:, nx:] = fgreen[ny:0:-1, nx:0:-1]

        self.fgreen = fgreen
        self.fgreentr = np.fft.fft2(fgreen).copy()
        
        if fftlib == 'pyfftw':
            try:
                import pyfftw
                print('Using PyFFTW')
                #prepare fftw's

                self.tmprho = (fgreen*(1.+1j))*0.
                fft_first = pyfftw.builders.fft(self.tmprho[:ny, :], axis = 1, threads = 1)

                self.tmpfft = (fgreen*(1.+1j))*0.
                self.tmpfft[:ny, :] = fft_first(self.tmprho[:ny, :])
                fft_second = pyfftw.builders.fft(self.tmpfft, axis = 0, threads = 1)
                
                self.phifft = fft_second(self.tmpfft) * self.fgreentr
                ifft_first = pyfftw.builders.ifft(self.phifft, axis = 0, threads = 1)

                self.tmpifft = ifft_first(self.phifft)
                ifft_second = pyfftw.builders.ifft(self.tmpifft[:ny, :], axis = 1, threads = 1)
                self.tmpphi  = (self.fgreen*(1.+1j))*0.

                #@profile
                def fft2_pyfftw():
                    self.tmpfft[:ny, :] = fft_first(self.tmprho[:ny, :])
                    self.phifft = fft_second(self.tmpfft) * self.fgreentr
                    self.tmpifft = ifft_first(self.phifft)
                    self.tmpphi[:ny, :] = ifft_second(self.tmpifft[:ny, :])


                self.fft2 = fft2_pyfftw

                self.tmpfft = (self.fgreen*(1.+1j))*0.
                self.tmpifft = (self.fgreen*(1.+1j))*0.
                
            except ImportError as err:
                print('Failed to import pyfftw')
                print('Got exception: ', err)

                fftlib = 'numpy'

        if fftlib == 'numpy':
            print('Using numpy FFT')

            self.tmprho = (self.fgreen*(1.+1j))*0.
            self.tmpphi = (self.fgreen*(1.+1j))*0.

            def fft2_numpy():
                self.phifft = np.fft.fft2(self.tmprho) * self.fgreentr
                self.tmpphi = np.fft.ifft2(self.phifft)

            self.fft2 = fft2_numpy
                
        elif fftlib != 'pyfftw':
            raise ValueError('fftlib not recognized!!!!')
            

        self.rho = np.zeros((self.Nxg,self.Nyg))
        self.phi = np.zeros((self.Nxg,self.Nyg))
        self.efx = np.zeros((self.Nxg,self.Nyg))
        self.efy = np.zeros((self.Nxg,self.Nyg))
        
        self.hlpphi = (self.phi*(1.+1j))*0.
        self.hlpefx = (self.efx*(1.+1j))*0.
        self.hlpefy = (self.efy*(1.+1j))*0.

        self.tmprho = (self.fgreen*(1.+1j))*0.
        self.tmpphi = (self.fgreen*(1.+1j))*0.

        self.Dh = Dh
        self.nx = nx
        self.ny = ny


    #@profile    
    def solve(self, rho = None, flag_verbose = False):
        if rho is None:
            rho = self.rho

        self._solve_core(rho)

        self.phi = np.real(self.hlpphi)
        self.efx = np.real(self.hlpefx)
        self.efy = np.real(self.hlpefy) 


    def get_state_object(self):
        state = PyPIC_Scatter_Gather(xg = self.xg, yg = self.yg)
        
        state.rho = self.rho.copy()
        state.phi = self.phi.copy()
        state.efx = self.efx.copy()
        state.efy = self.efy.copy()
        
        return state

        
    #~ def load_state_object(self, state):
        #~ self.rho = state.rho.copy()
        #~ self.phi = state.phi.copy()
        #~ self.efx = state.efx.copy()
        #~ self.efy = state.efy.copy()

        
    #@profile
    def solve_states(self, states):
        
        states = np.atleast_1d(states)

        if len(states) > 2:
            raise ValueError('Not implemented yet! Sorry.')

        elif len(states) == 1:

            state = states[0]
            
            self._solve_core(state.rho)
            
            state.phi = np.real(self.hlpphi)
            state.efx = np.real(self.hlpefx)
            state.efy = np.real(self.hlpefy) 

        else:
            rho = 1*states[0].rho + 1j*states[1].rho
            
            self._solve_core(rho)
            
            states[1].phi = np.imag(self.hlpphi)
            states[1].efx = np.imag(self.hlpefx)
            states[1].efy = np.imag(self.hlpefy)
            states[0].phi = np.real(self.hlpphi)
            states[0].efx = np.real(self.hlpefx)
            states[0].efy = np.real(self.hlpefy)

    #@profile
    def _solve_core(self, rho):

        self.tmprho[:self.ny, :self.nx] = rho.T

        self.fft2()

        self.hlpphi = 1./(4. * np.pi * eps0)*(self.tmpphi[:self.ny, :self.nx]).T

        self.hlpefx[1:self.Nxg-1,:] = self.hlpphi[0:self.Nxg-2,:] - self.hlpphi[2:self.Nxg,:];  #central difference on internal nodes
        self.hlpefy[:,1:self.Nyg-1] = self.hlpphi[:,0:self.Nyg-2] - self.hlpphi[:,2:self.Nyg];  #central difference on internal nodes
        
        self.hlpefx = self.hlpefx/(2*self.dx)
        self.hlpefy = self.hlpefy/(2*self.dy)



