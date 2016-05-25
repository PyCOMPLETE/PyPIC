import numpy as np
import FiniteDifferences_Staircase_SquareGrid as PIC_FD
import simple_polygon as spoly
from PyPIC_Scatter_Gather import PyPIC_Scatter_Gather

from scipy.constants import e as qe

class AddInternalGrid(PyPIC_Scatter_Gather):
    def __init__(self, pic_external, x_min_internal, x_max_internal, y_min_internal, y_max_internal, Dh_internal, N_nodes_discard):
        
        #build boundary for refinement grid
        box_internal = spoly.SimplePolygon({'Vx':np.array([x_max_internal, x_min_internal, x_min_internal, x_max_internal]),
                                'Vy':np.array([y_max_internal, y_max_internal, y_min_internal, y_min_internal])})
        self.pic_internal = PIC_FD.FiniteDifferences_Staircase_SquareGrid(chamb = box_internal, Dh = Dh_internal, remove_external_nodes_from_mat=False)

        
        self.pic_external = pic_external
        
        self.x_min_internal = x_min_internal
        self.x_max_internal = x_max_internal
        self.y_min_internal = y_min_internal
        self.y_max_internal = y_max_internal
        self.Dh_internal = Dh_internal
        self.N_nodes_discard = N_nodes_discard
        self.D_discard = N_nodes_discard*Dh_internal
        
    
    def scatter(self, x_mp, y_mp, nel_mp, charge = -qe):
        self.pic_external.scatter(x_mp, y_mp, nel_mp, charge)
        self.pic_internal.scatter(x_mp, y_mp, nel_mp, charge)

         
    def gather(self, x_mp, y_mp):
        mask_internal = np.logical_and(\
            np.logical_and(x_mp > self.x_min_internal + self.D_discard, 
                           x_mp < self.x_max_internal - self.D_discard),
            np.logical_and(y_mp > self.y_min_internal + self.D_discard, 
                           y_mp < self.y_max_internal - self.D_discard))
                           
        mask_external = np.logical_not(mask_internal)
        
        Ex_sc_n_external, Ey_sc_n_external = self.pic_external.gather(x_mp[mask_external], y_mp[mask_external])
        Ex_sc_n_internal, Ey_sc_n_internal = self.pic_internal.gather(x_mp[mask_internal], y_mp[mask_internal])
        
        Ex_sc_n = 0.*x_mp
        Ey_sc_n = 0.*x_mp
        
        Ex_sc_n[mask_external] = Ex_sc_n_external
        Ey_sc_n[mask_external] = Ey_sc_n_external
        Ex_sc_n[mask_internal] = Ex_sc_n_internal
        Ey_sc_n[mask_internal] = Ey_sc_n_internal
        
        return Ex_sc_n, Ey_sc_n
        
    def gather_phi(self, x_mp, y_mp):
        mask_internal = np.logical_and(\
            np.logical_and(x_mp > self.x_min_internal + self.D_discard, 
                           x_mp < self.x_max_internal - self.D_discard),
            np.logical_and(y_mp > self.y_min_internal + self.D_discard, 
                           y_mp < self.y_max_internal - self.D_discard))
                           
        mask_external = np.logical_not(mask_internal)
        
        phi_sc_n_external = self.pic_external.gather_phi(x_mp[mask_external], y_mp[mask_external])
        phi_sc_n_internal = self.pic_internal.gather_phi(x_mp[mask_internal], y_mp[mask_internal])
        
        phi_sc_n = 0.*x_mp
        
        phi_sc_n[mask_external] = phi_sc_n_external
        phi_sc_n[mask_internal] = phi_sc_n_internal

        return phi_sc_n

    def solve(self, rho = None, flag_verbose = False):
        if rho is not None:
            raise ValueError('rho matrix cannot be provided in multigrid mode!')
        self.pic_external.solve(flag_verbose = flag_verbose)
        self.pic_internal.solve(flag_verbose = flag_verbose, pic_external=self.pic_external)
