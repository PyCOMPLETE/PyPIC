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
#		           PyPIC Version 1.00                     
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

qe = e
eps0 = epsilon_0

class FiniteDifferences_Staircase_SquareGrid(PyPIC_Scatter_Gather):
    #@profile
    def __init__(self,chamb, Dh, sparse_solver = 'scipy_slu'):
        
		print 'Start PIC init.:'
		print 'Finite Differences, Square Grid'


		super(FiniteDifferences_Staircase_SquareGrid, self).__init__(chamb.x_aper, chamb.y_aper, Dh)
		Nyg, Nxg = self.Nyg, self.Nxg
		
		
		[xn, yn]=np.meshgrid(self.xg,self.yg)

		xn=xn.T
		xn=xn.flatten()

		yn=yn.T
		yn=yn.flatten()
		#% xn and yn are stored such that the external index is on x 

		flag_outside_n=chamb.is_outside(xn,yn)
		flag_inside_n=~(flag_outside_n)


		flag_outside_n_mat=np.reshape(flag_outside_n,(Nyg,Nxg),'F');
		flag_outside_n_mat=flag_outside_n_mat.T
		[gx,gy]=np.gradient(np.double(flag_outside_n_mat));
		gradmod=abs(gx)+abs(gy);
		flag_border_mat=np.logical_and((gradmod>0), flag_outside_n_mat);
		flag_border_n = flag_border_mat.flatten()

		A=scsp.lil_matrix((Nxg*Nyg,Nxg*Nyg)); #allocate a sparse matrix

		list_internal_force_zero = []

		# Build A matrix
		for u in range(0,Nxg*Nyg):
			if np.mod(u, Nxg*Nyg/20)==0:
				print ('Mat. assembly %.0f'%(float(u)/ float(Nxg*Nyg)*100)+"""%""")
			if flag_inside_n[u]:
				A[u,u] = -(4./(Dh*Dh))
				A[u,u-1]=1./(Dh*Dh);     #phi(i-1,j)nx
				A[u,u+1]=1./(Dh*Dh);     #phi(i+1,j)
				A[u,u-Nyg]=1./(Dh*Dh);    #phi(i,j-1)
				A[u,u+Nyg]=1./(Dh*Dh);    #phi(i,j+1)
			else:
				# external nodes
				A[u,u]=1.
				
		A=A.tocsr() #convert to csr format
		
		#Remove trivial equtions 
		diagonal = A.diagonal()
		N_full = len(diagonal)
		indices_non_id = np.where(diagonal!=1.)[0]
		N_sel = len(indices_non_id)

		Msel = scsp.lil_matrix((N_full, N_sel))
		for ii, ind in enumerate(indices_non_id):
			Msel[ind, ii] =1.
			
		Msel = Msel.tocsc()

		Asel = Msel.T*A*Msel
		Asel=Asel.tocsc()
		

		if sparse_solver == 'scipy_slu':
			print "Using scipy superlu solver..."
			luobj = ssl.splu(Asel.tocsc())
		elif sparse_solver == 'PyKLU':
			print "Using klu solver..."
			try:
				import PyKLU.klu as klu
				luobj = klu.Klu(Asel.tocsc())
			except StandardError, e: 
				print "Got exception: ", e
				print "Falling back on scipy superlu solver:"
				luobj = ssl.splu(Asel.tocsc())
		else:
			raise ValueError('Solver not recognized!!!!\nsparse_solver must be "scipy_klu" or "PyKLU"\n')
			
		self.xn = xn
		self.yn = yn
		
		self.flag_inside_n = flag_inside_n
		self.flag_outside_n = flag_outside_n
		self.flag_outside_n_mat = flag_outside_n_mat
		self.flag_inside_n_mat = np.logical_not(flag_outside_n_mat)
		self.flag_border_mat = flag_border_mat
		self.Asel = Asel
		self.luobj = luobj


		self.rho = np.zeros((self.Nxg,self.Nyg));
		self.phi = np.zeros((self.Nxg,self.Nyg));
		self.efx = np.zeros((self.Nxg,self.Nyg));
		self.efy = np.zeros((self.Nxg,self.Nyg));

		
		self.U_sc_eV_stp=0.;

		
		self.Msel = Msel.tocsc()
		self.Msel_T = (Msel.T).tocsc()

		
		print 'Done PIC init.'
                        

    #@profile    
    def solve(self, rho = None, flag_verbose = False):

		if rho == None:
			rho = self.rho

		b=-rho.flatten()/eps0;
		b[~(self.flag_inside_n)]=0.; #boundary condition

		if flag_verbose:
			print 'Start Linear System Solution.'
		b_sel = self.Msel_T*b
		phi_sel = self.luobj.solve(b_sel)
		phi = self.Msel*phi_sel

		U_sc_eV_stp = -0.5*eps0*np.sum(b*phi)*self.Dh*self.Dh/qe

		if flag_verbose:
			print 'Start field computation.'


		phi=np.reshape(phi,(self.Nxg,self.Nyg))

		efx = self.efx
		efy = self.efy

		efx[1:self.Nxg-1,:] = phi[0:self.Nxg-2,:] - phi[2:self.Nxg,:];  #central difference on internal nodes
		efy[:,1:self.Nyg-1] = phi[:,0:self.Nyg-2] - phi[:,2:self.Nyg];  #central difference on internal nodes

		efx[self.flag_border_mat]=efx[self.flag_border_mat]*2;
		efy[self.flag_border_mat]=efy[self.flag_border_mat]*2;
		
		self.efx = efx / (2*self.Dh);    #divide grid size
		self.efy = efy / (2*self.Dh);

		 
		self.rho = rho
		self.b = b
		self.phi = phi
		self.U_sc_eV_stp = U_sc_eV_stp
        


