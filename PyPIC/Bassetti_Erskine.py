#----------------------------------------------------------------------
#                                                                      
#                           CERN                                       
#                                                                      
#     European Organization for Nuclear Research                       
#                                                                      
#     
#     This file is part of the code
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
from .errffor import errf


qe = e
eps0 = epsilon_0

class Interpolated_Bassetti_Erskine(PyPIC_Scatter_Gather):
    #@profile
    def __init__(self, x_aper, y_aper, Dh, sigmax, sigmay, 
        n_imag_ellip=0, tot_charge=1., verbose=True, allow_scatter_and_solve=False):
        
        self.verbose = verbose
        self.allow_scatter_and_solve = allow_scatter_and_solve
        
        if self.verbose:
            print('Start PIC init.:')
            print('Bassetti-Erskine, Square Grid')

        self.Dh = Dh
        super(Interpolated_Bassetti_Erskine, self).__init__(x_aper, y_aper, self.Dh, self.Dh, verbose=self.verbose)
        
        xx = self.xg
        yy = self.yg
        
        Ex=np.zeros((len(xx),len(yy)),dtype=complex);
        Ey=np.zeros((len(xx),len(yy)),dtype=complex);
        
        for ii in range(len(xx)):

            if np.mod(ii, len(xx)//20)==0 and self.verbose:
                print(('Bassetti Erskine evaluation %.0f'%(float(ii)/ float(len(xx))*100)+"""%"""))

            for jj in range(len(yy)):
                x=xx[ii];
                y=yy[jj];
                Ex_imag,Ey_imag  = ImageTerms(x,y,x_aper,y_aper,0,0, n_imag_ellip)
                Ex_BE,Ey_BE      = BassErsk(x,y,sigmax,sigmay)
                Ex[ii,jj] = Ex_BE + Ex_imag
                Ey[ii,jj] = Ey_BE + Ey_imag
                
        YY,XX = np.meshgrid(self.yg, self.xg)		
        self.rho = tot_charge/(2.*np.pi*sigmax*sigmay)*np.exp(-(XX)**2/(2.*sigmax**2)-(YY)**2/(2.*sigmay**2))
        self.phi = np.zeros((self.Nxg,self.Nyg))
        self.efx = tot_charge * Ex.real
        self.efy = tot_charge * Ey.real
                        

    #@profile    
    def solve(self, rho = None, flag_verbose = False):
        if not self.allow_scatter_and_solve:
            raise ValueError('Bassetti_Erskine: nothing to solve!!!!')
        
    def scatter(self,  x_mp, y_mp, nel_mp, charge = -qe, flag_add=False):
        if not self.allow_scatter_and_solve:
            raise ValueError('Bassetti_Erskine: what do you want to scatter???!!!!')
        
        


def wfun(z):
    x=z.real
    y=z.imag
    wx,wy=errf(x,y)
    return wx+1j*wy

def BassErsk(xin,yin,sigmax,sigmay):
        
    x=abs(xin);
    y=abs(yin);
    
    
    
    if sigmax>sigmay:
    
        S=np.sqrt(2*(sigmax*sigmax-sigmay*sigmay));
        factBE=1/(2*eps0*np.sqrt(np.pi)*S);
        etaBE=sigmay/sigmax*x+1j*sigmax/sigmay*y;
        zetaBE=x+1j*y;
        
        val=factBE*(wfun(zetaBE/S)-np.exp( -x*x/(2*sigmax*sigmax)-y*y/(2*sigmay*sigmay))*wfun(etaBE/S) );
           
        Ex=abs(val.imag)*np.sign(xin);
        Ey=abs(val.real)*np.sign(yin);
    
    else:
    
        S=np.sqrt(2*(sigmay*sigmay-sigmax*sigmax));
        factBE=1/(2*eps0*np.sqrt(np.pi)*S);
        etaBE=sigmax/sigmay*y+1j*sigmay/sigmax*x;
        yetaBE=y+1j*x;
        
        val=factBE*(wfun(yetaBE/S)-np.exp( -y*y/(2*sigmay*sigmay)-x*x/(2*sigmax*sigmax))*wfun(etaBE/S) );
           
        Ey=abs(val.imag)*np.sign(yin);
        Ex=abs(val.real)*np.sign(xin);
         
    return Ex, Ey

def ImageTerms(x,y,a,b,x0,y0, nimag):
    
        
    eps0=epsilon_0;    
    
    if nimag>0 and abs((a-b)/a)>1e-3:    
        g=np.sqrt(a*a-b*b)
        z=x+1j*y
        q=np.arccosh(z/g)
        mu=q.real
        phi=q.imag
        
        z0=x0+1j*y0
        q0=np.arccosh(z0/g)
        mu0=q0.real
        phi0=q0.imag
        
        mu1=0.5*np.log((a+b)/(a-b))
        
        Ecpx=0+0j
        
        
        q=np.conj(q)
        for nn in range(1,nimag+1):
            Ecpx=Ecpx+np.exp(-nn*mu1) * ( (np.cosh(nn*mu0)*np.cos(nn*phi0)) / (np.cosh(nn*mu1)) + 1j * (np.sinh(nn*mu0)*np.sin(nn*phi0)) / (np.sinh(nn*mu1))   )* (np.sinh(nn*q))/(np.sinh(q))
            
        
        Ecpx=Ecpx/(4*np.pi*eps0)*4/g
        Ex=Ecpx.real
        Ey=Ecpx.imag
    else:
        if (x0==0) and (y0==0):
            Ex=0.
            Ey=0.
        else:
            print('This case has not been implemented yet')
    
    return Ex, Ey


