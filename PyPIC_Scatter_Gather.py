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
from . import rhocompute as rhocom
from . import int_field_for as iff
#~ from abc import abstractmethod, ABCMeta

na = lambda x:np.array([x])

qe=1.602176565e-19;
eps0=8.8541878176e-12;

class PyPIC_Scatter_Gather(object):
    #__metadata__ = ABCMeta

    def __init__(self, x_aper=None, y_aper=None, dx=None, dy=None, xg=None, yg=None, 
                x_min=None, x_max=None, y_min=None, y_max=None, *args, **kwargs):

        print('PyPIC Version 2.4.5')
        
        if xg is not None and yg is not None:
            assert(x_aper is None and y_aper is None and dx is None and dy is None)
            assert(x_min is None and x_max is None and y_min is None and y_max is None)

            Nxg=len(xg);
            bias_x=min(xg);

            Nyg=len(yg);
            bias_y=min(yg);
            
            dx = xg[1]-xg[0]
            dy = yg[1]-yg[0]

        elif dx is not None and dy is not None:
            assert(xg is None and yg is None)
            # box given
            if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
                assert(x_aper is None and y_aper is None)

                x_aper = (x_max-x_min)/2.
                x_center = (x_max+x_min)/2.

                y_aper = (y_max-y_min)/2.
                y_center = (y_max+y_min)/2.
            # aperture given
            elif x_aper is not None and y_aper is not None:
                assert(x_min is None and x_max is None and y_min is None and y_max is None)

                x_center = 0.
                y_center = 0.

            else:
                raise ValueError('x_aper and y_aper, or x_min, x_max and y_min, y_max must be specified!!!')

            xg=np.arange(0, x_aper+5.*dx,dx,float)  
            xgr=xg[1:]
            xgr=xgr[::-1]#reverse array
            xg=np.concatenate((-xgr,xg),0)
            xg = xg + x_center
            Nxg=len(xg);
            bias_x=min(xg);

            yg=np.arange(0,y_aper+4.*dy,dy,float)  
            ygr=yg[1:]
            ygr=ygr[::-1]#reverse array
            yg=np.concatenate((-ygr,yg),0)
            yg = yg + y_center
            Nyg=len(yg);
            bias_y=min(yg);	

        else:
            raise ValueError('dx and dy, or xg and yg must be specified!!!')


        self.dx = dx
        self.xg = xg
        self.Nxg = Nxg
        self.bias_x = bias_x
        self.dy = dy
        self.yg = yg
        self.Nyg = Nyg
        self.bias_y = bias_y

                        
    #@profile
    def scatter(self, x_mp, y_mp, nel_mp, charge = -qe, flag_add=False):
        
        if not (len(x_mp)==len(y_mp)==len(nel_mp)):
            raise ValueError('x_mp, y_mp, nel_mp should have the same length!!!')
        
        if len(x_mp)>0:
            rho=rhocom.compute_sc_rho(x_mp,y_mp,nel_mp,self.bias_x,self.bias_y,self.dx,self.dy,self.Nxg,self.Nyg)
        else:
            rho=self.rho*0.

        if flag_add:
            self.rho+=charge*rho/(self.dx*self.dy);
        else:
            self.rho=charge*rho/(self.dx*self.dy);

         
    def gather(self, x_mp, y_mp):
        
        if not (len(x_mp)==len(y_mp)):
            raise ValueError('x_mp, y_mp should have the same length!!!')

        if len(x_mp)>0:    
            ## compute beam electric field
            Ex_sc_n, Ey_sc_n = iff.int_field(x_mp,y_mp,self.bias_x,self.bias_y,self.dx,
                                         self.dy, self.efx, self.efy)
                       
        else:
            Ex_sc_n=0.
            Ey_sc_n=0.
            
        return Ex_sc_n, Ey_sc_n
        
    def gather_phi(self, x_mp, y_mp):
        
        if not (len(x_mp)==len(y_mp)):
            raise ValueError('x_mp, y_mp should have the same length!!!')

        if len(x_mp)>0:    
            ## compute beam potential
            phi_sc_n, _ = iff.int_field(x_mp,y_mp,self.bias_x,self.bias_y,self.dx,
                                         self.dy, self.phi, self.phi)
                       
        else:
            phi_sc_n=0.
            
        return phi_sc_n
        
    def gather_rho(self, x_mp, y_mp):
        
        if not (len(x_mp)==len(y_mp)):
            raise ValueError('x_mp, y_mp should have the same length!!!')

        if len(x_mp)>0:    
            ## compute beam distribution
            rho_sc_n, _ = iff.int_field(x_mp,y_mp,self.bias_x,self.bias_y,self.dx,
                                         self.dy, self.rho, self.rho)
                       
        else:
            rho_sc_n=0.
            
        return rho_sc_n

    #@abstractmethod
    def solve(self, *args, **kwargs):
        '''Computes the electric field maps from the stored 
        charge distribution (self.rho) and stores them in
        self.efx, self.efy.'''
        pass
        
    #@profile
    def scatter_and_solve(self, x_mp, y_mp, nel_mp, charge = -qe, flag_add=False):
        self.scatter(x_mp, y_mp, nel_mp, charge, flag_add)
        self.solve()


    def _solve_for_states(self,*args, **kwargs):
        raise ValueError('I am a state, I cannot solve!')
