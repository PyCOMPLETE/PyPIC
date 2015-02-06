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
#		           PyPIC Version 0.00                     
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

class FFT_OpenBoundary_SquareGrid(PyPIC_Scatter_Gather):
    #@profile
    def __init__(self, x_aper, y_aper, Dh):
        
		print 'Start PIC init.:'
		print 'FFT, Open Boundary, Square Grid'


		super(FFT_OpenBoundary_SquareGrid, self).__init__(x_aper, y_aper, Dh)

		
		dx = self.Dh
		dy = self.Dh
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
		
		self.fgreen = np.fft.fft2(fgreen)
		self.rho = np.zeros((self.Nxg,self.Nyg))
		self.phi = np.zeros((self.Nxg,self.Nyg))
		self.efx = np.zeros((self.Nxg,self.Nyg))
		self.efy = np.zeros((self.Nxg,self.Nyg))
		
		self.dx = dx
		self.dy = dy
		self.nx = nx
		self.ny = ny
                        

    #@profile    
    def solve(self, rho = None, flag_verbose = False):
		if rho == None:
			rho = self.rho

		tmprho = 0.*self.fgreen
		tmprho[:self.ny, :self.nx] = rho.T

		fftphi = np.fft.fft2(tmprho) * self.fgreen

		tmpphi = np.fft.ifft2(fftphi)
		self.phi = 1./(4. * np.pi * eps0)*np.real(tmpphi[:self.ny, :self.nx]).T

		self.efx[1:self.Nxg-1,:] = self.phi[0:self.Nxg-2,:] - self.phi[2:self.Nxg,:];  #central difference on internal nodes
		self.efy[:,1:self.Nyg-1] = self.phi[:,0:self.Nyg-2] - self.phi[:,2:self.Nyg];  #central difference on internal nodes

		
		self.efy = self.efy/(2*self.Dh)
		self.efx = self.efx/(2*self.Dh)
        
        



