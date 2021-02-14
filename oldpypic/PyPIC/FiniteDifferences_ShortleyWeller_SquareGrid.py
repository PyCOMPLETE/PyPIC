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

from . import int_field_for_border as iffb


na = lambda x:np.array([x])

qe = e
eps0 = epsilon_0

class FiniteDifferences_ShortleyWeller_SquareGrid(PyPIC_Scatter_Gather):
    #@profile
    def __init__(self,chamb, Dh, sparse_solver = 'scipy_slu', tol_stem = 0.01, tol_der = 0.1, include_solver=True):

        print('Start PIC init.:')
        print('Finite Differences, Shortley-Weller, Square Grid')
        print('Using Shortley-Weller boundary approx.')

        self.Dh = Dh
        super(FiniteDifferences_ShortleyWeller_SquareGrid, self).__init__(chamb.x_aper, chamb.y_aper, self.Dh, self.Dh)
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


        if include_solver:
            A=scsp.lil_matrix((Nxg*Nyg,Nxg*Nyg)); #allocate a sparse matrix
            Dx=scsp.lil_matrix((Nxg*Nyg,Nxg*Nyg)); #allocate a sparse matrix
            Dy=scsp.lil_matrix((Nxg*Nyg,Nxg*Nyg)); #allocate a sparse matrix

            list_internal_force_zero = []
            # Build A Dx Dy matrices 
            for u in range(0,Nxg*Nyg):
                if np.mod(u, Nxg*Nyg//20)==0:
                    print(('Mat. assembly %.0f'%(float(u)/ float(Nxg*Nyg)*100)+"""%"""))
                if flag_inside_n[u]:

                    #Compute Shortley-Weller coefficients
                    if flag_inside_n[u-1]: #phi(i-1,j)
                        hw = Dh
                    else:
                        x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u-1]), na(yn[u-1]), na(0.), resc_fac=.995, flag_robust=False)
                        hw = np.abs(y_int[0]-yn[u])

                    if flag_inside_n[u+1]: #phi(i+1,j)
                        he = Dh
                    else:
                        x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u+1]), na(yn[u+1]), na(0.), resc_fac=.995, flag_robust=False)
                        he = np.abs(y_int[0]-yn[u])

                    if flag_inside_n[u-Nyg]: #phi(i,j-1)
                        hs = Dh
                    else:
                        x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u-Nyg]), na(yn[u-Nyg]), na(0.), resc_fac=.995, flag_robust=False)
                        hs = np.abs(x_int[0]-xn[u])
                        #~ print hs

                    if flag_inside_n[u+Nyg]: #phi(i,j+1)
                        hn = Dh
                    else:
                        x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u+Nyg]), na(yn[u+Nyg]), na(0.), resc_fac=.995, flag_robust=False)
                        hn = np.abs(x_int[0]-xn[u])
                        #~ print hn


                    # Build A matrix
                    if hn<Dh*tol_stem or hs<Dh*tol_stem or hw<Dh*tol_stem or he<Dh*tol_stem: # nodes very close to the bounday
                        A[u,u] =1.
                        list_internal_force_zero.append(u)
                        #print u, xn[u], yn[u]
                    else:
                        A[u,u] = -(2./(he*hw)+2/(hs*hn))
                        A[u,u-1]=2./(hw*(hw+he));     #phi(i-1,j)nx
                        A[u,u+1]=2./(he*(hw+he));     #phi(i+1,j)
                        A[u,u-Nyg]=2./(hs*(hs+hn));    #phi(i,j-1)
                        A[u,u+Nyg]=2./(hn*(hs+hn));    #phi(i,j+1)

                    # Build Dx matrix
                    if hn<Dh*tol_der:
                        if hs>=Dh*tol_der:
                            Dx[u,u] = -1./hs
                            Dx[u,u-Nyg]=1./hs
                    elif hs<Dh*tol_der:
                        if hn>=Dh*tol_der:
                            Dx[u,u] = 1./hn
                            Dx[u,u+Nyg]=-1./hn
                    else:
                        Dx[u,u] = (1./(2*hn)-1./(2*hs))
                        Dx[u,u-Nyg]=1./(2*hs)
                        Dx[u,u+Nyg]=-1./(2*hn)


                    # Build Dy matrix	
                    if he<Dh*tol_der:
                        if hw>=Dh*tol_der:
                            Dy[u,u] = -1./hw
                            Dy[u,u-1]=1./hw
                    elif hw<Dh*tol_der:
                        if he>=Dh*tol_der:
                            Dy[u,u] = 1./he
                            Dy[u,u+1]=-1./(he)
                    else:
                        Dy[u,u] = (1./(2*he)-1./(2*hw))
                        Dy[u,u-1]=1./(2*hw)
                        Dy[u,u+1]=-1./(2*he)

                else:
                    # external nodes
                    A[u,u]=1.


            flag_force_zero = flag_outside_n.copy()
            for ind in list_internal_force_zero:
                flag_force_zero[ind] = True

            flag_force_zero_mat=np.reshape(flag_force_zero,(Nyg,Nxg),'F');
            flag_force_zero_mat=flag_force_zero_mat.T

            print('Internal nodes with 0 potential')
            print(list_internal_force_zero)

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


            self.xn = xn
            self.yn = yn

            self.flag_inside_n = flag_inside_n
            self.flag_outside_n = flag_outside_n
            self.flag_outside_n_mat = flag_outside_n_mat
            self.flag_force_zero = flag_force_zero
            self.Asel = Asel

            self.Dx = Dx.tocsc()

            self.Dy = Dy.tocsc()

            self.sparse_solver = sparse_solver

            self.U_sc_eV_stp=0.;


            self.Msel = Msel.tocsc()
            self.Msel_T = (Msel.T).tocsc()

            #initialize self.luobj
            self.build_sparse_solver()


            self.tol_der = tol_der
            self.tol_stem = tol_stem

            print('Done PIC init.')

        else:

            self.solve = self._solve_for_states
            self.sparse_solver = None
            self.tol_stem = None
            self.tol_der = None

        self.flag_inside_n_mat = np.logical_not(flag_outside_n_mat)
        self.chamb = chamb
        self.rho = np.zeros((self.Nxg,self.Nyg));
        self.phi = np.zeros((self.Nxg,self.Nyg));
        self.efx = np.zeros((self.Nxg,self.Nyg));
        self.efy = np.zeros((self.Nxg,self.Nyg));

    #@profile    
    def solve(self, rho = None, flag_verbose = False):

        if rho is None:
            rho = self.rho
        self._solve_core(self, rho)


    def gather(self, x_mp, y_mp):

        if not (len(x_mp)==len(y_mp)):
            raise ValueError('x_mp, y_mp should have the same length!!!')

        if len(x_mp)>0:
            ## compute beam electric field
            Ex_sc_n, Ey_sc_n = iffb.int_field_border(x_mp,y_mp,self.bias_x,self.bias_y,self.Dh,
                                         self.Dh, self.efx, self.efy, self.flag_inside_n_mat)

        else:
            Ex_sc_n=0.
            Ey_sc_n=0.

        return Ex_sc_n, Ey_sc_n

    def build_sparse_solver(self):

        if self.sparse_solver == 'scipy_slu':
            print("Using scipy superlu solver...")
            luobj = ssl.splu(self.Asel.tocsc())
        elif self.sparse_solver == 'PyKLU':
            print("Using klu solver...")
            try:
                import PyKLU.klu as klu
                luobj = klu.Klu(self.Asel.tocsc())
            except Exception as e:
                print("Got exception: ", e)
                print("Falling back on scipy superlu solver:")
                luobj = ssl.splu(self.Asel.tocsc())
        else:
            raise ValueError('Solver not recognized!!!!\nsparse_solver must be "scipy_slu" or "PyKLU"\n')

        self.luobj = luobj

    def get_state_object(self):
        state = FiniteDifferences_ShortleyWeller_SquareGrid(chamb=self.chamb, Dh=self.Dh,
                    sparse_solver = self.sparse_solver, tol_stem = self.tol_stem, tol_der = self.tol_der,
                    include_solver=False)

        state.rho = self.rho.copy()
        state.phi = self.phi.copy()
        state.efx = self.efx.copy()
        state.efy = self.efy.copy()

        return state


    def solve_states(self, states):

        states = np.atleast_1d(states)
        for ii in range(len(states)):
            state = states[ii]
            self._solve_core(state, state.rho)


    def _solve_core(self, state, rho):

        b=-rho.flatten()/eps0;
        b[(self.flag_force_zero)]=0;
        b_sel = self.Msel_T*b
        phi_sel = self.luobj.solve(b_sel)
        phi = self.Msel*phi_sel

        efx = self.Dx*phi
        efy = self.Dy*phi
        phi=np.reshape(phi,(self.Nxg,self.Nyg))
        efx=np.reshape(efx,(self.Nxg,self.Nyg))
        efy=np.reshape(efy,(self.Nxg,self.Nyg))
        state.efx = efx
        state.efy = efy
        state.phi = phi


