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
from .PyPIC_Scatter_Gather import PyPIC_Scatter_Gather
from scipy.constants import e, epsilon_0
import scipy as sp

na = lambda x:np.array([x])


qe=e
eps0=epsilon_0



    

class FFT_PEC_Boundary_SquareGrid(PyPIC_Scatter_Gather):
    #@profile
    def __init__(self, x_aper, y_aper, Dh, fftlib='pyfftw'):
        
        print('Start PIC init.:')
        print('FFT, PEC Boundary, Square Grid')


        self.Dh = Dh		
        super(FFT_PEC_Boundary_SquareGrid, self).__init__(x_aper, y_aper, self.Dh, self.Dh)

        
        
        self.i_min = np.min(np.where(self.xg>=-x_aper)[0])
        self.i_max = np.max(np.where(self.xg<=x_aper)[0])+1
        self.j_min = np.min(np.where(self.yg>=-y_aper)[0])
        self.j_max = np.max(np.where(self.yg<=y_aper)[0])+1

        self.rho = np.zeros((self.Nxg,self.Nyg))
        self.phi = np.zeros((self.Nxg,self.Nyg))
        self.efx = np.zeros((self.Nxg,self.Nyg))
        self.efy = np.zeros((self.Nxg,self.Nyg))
        
        
        m, n = self.rho[self.i_min:self.i_max,self.j_min:self.j_max].shape;

        xx = np.arange(1,m+0.5,1);
        yy = np.arange(1,n+0.5,1);
        
        YY, XX = np.meshgrid(yy,xx) 
        self.green = 4.*eps0*(np.sin(XX/2*np.pi/float(m+1.))**2/self.Dh**2+\
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
                rhocut = self.rho[self.i_min:self.i_max,self.j_min:self.j_max]
                m, n = rhocut.shape;
                tmp = np.zeros((2*m + 2, n))
                self.ffti = pyfftw.builders.fft(tmp.copy(), axis=0)
                tmp = np.zeros((m, 2*n + 2))
                self.fftj = pyfftw.builders.fft(tmp.copy(), axis=1)
            except ImportError as err:
                print('Failed to import pyfftw')
                print('Got exception: ', err)
                print('Using numpy fft')
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


    #@profile    
    def solve(self, rho = None, flag_verbose = False):
        if rho == None:
            rho = self.rho

        rhocut = rho[self.i_min:self.i_max,self.j_min:self.j_max]
        
        rho_bar =  self.dst2(rhocut)       
        phi_bar = rho_bar/self.green    
        self.phi[self.i_min:self.i_max,self.j_min:self.j_max] = self.dst2(phi_bar).copy()

        
        self.efx[1:self.Nxg-1,:] = self.phi[0:self.Nxg-2,:] - self.phi[2:self.Nxg,:];  #central difference on internal nodes
        self.efy[:,1:self.Nyg-1] = self.phi[:,0:self.Nyg-2] - self.phi[:,2:self.Nyg];  #central difference on internal nodes

        self.efx[self.flag_border_mat]=self.efx[self.flag_border_mat]*2;
        self.efy[self.flag_border_mat]=self.efy[self.flag_border_mat]*2;
        
        
        self.efy = self.efy/(2*self.Dh)
        self.efx = self.efx/(2*self.Dh)
        
        



