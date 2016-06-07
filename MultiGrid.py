import numpy as np
import FiniteDifferences_Staircase_SquareGrid as PIC_FD
import FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
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
        self.chamb = self.pic_external.chamb
        
        self.x_min_internal = x_min_internal
        self.x_max_internal = x_max_internal
        self.y_min_internal = y_min_internal
        self.y_max_internal = y_max_internal
        self.Dh_internal = Dh_internal
        self.N_nodes_discard = N_nodes_discard
        self.D_discard = N_nodes_discard*Dh_internal
        
        #check if the internal grid lies inside the chamber
        x_border = self.pic_internal.xn[self.pic_internal.flag_border_n]
        y_border = self.pic_internal.yn[self.pic_internal.flag_border_n]
        if pic_external.chamb.is_outside(x_border, y_border).any() == True:
            raise ValueError('The internal grid is outside the chamber!')
            
    
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
        
        
class AddMultiGrids(PyPIC_Scatter_Gather):
    def __init__(self, pic_main, grids):

        n_grids = len(grids)
        pic_list = [pic_main]
        for ii in xrange(n_grids):
            x_min_internal = grids[ii]['x_min_internal']
            x_max_internal = grids[ii]['x_max_internal']
            y_min_internal = grids[ii]['y_min_internal']
            y_max_internal = grids[ii]['y_max_internal']
            Dh_internal = grids[ii]['Dh_internal']
            N_nodes_discard = grids[ii]['N_nodes_discard']
            pic_list.append(AddInternalGrid(pic_list[-1], x_min_internal, x_max_internal, y_min_internal, 
                            y_max_internal, Dh_internal, N_nodes_discard))
                            
        pic_list = pic_list[1:]                    
        self.grids = grids          
        self.pic_list = pic_list
        
        self.scatter = self.pic_list[-1].scatter
        self.solve = self.pic_list[-1].solve
        self.gather = self.pic_list[-1].gather
        self.gather_phi = self.pic_list[-1].gather_phi
        
        
        
class AddTelescopicGrids(PyPIC_Scatter_Gather):
    def __init__(self, pic_main, f_telescope, target):
    
        x_min_target = target['x_min_target']
        x_max_target = target['x_max_target']
        y_min_target = target['y_min_target']
        y_max_target = target['y_max_target']
        Dh_target = target['Dh_target']
        N_nodes_discard = target['N_nodes_discard']    
        N_min_Dh = target['N_min_Dh']    
        Dh_main = target['Dh_main']
        
        x_center_target = (x_min_target + x_max_target)/2.
        y_center_target = (y_min_target + y_max_target)/2.
        
        Sx_target = x_max_target - x_min_target
        Sy_target = y_max_target - y_min_target
        
        if Sx_target < Sy_target:
            S_target = Sx_target 
        else:
            S_target = Sy_target 
            
        if f_telescope <= 0. or f_telescope >=1.:
			raise ValueError('The magnification factor between grids must be 0<f<1!!!')    
        n_grids = int(np.ceil(np.log(S_target/(N_min_Dh*Dh_main))/np.log(f_telescope)))+1
        if n_grids <= 0.:
            raise ValueError('Found number of grids = %d <= 0!!!'%n_grids)
        
        print '%d grids needed'%n_grids
        
        if n_grids == 1:
            f_exact = None #it's not used
        else:
            f_exact = (S_target/(N_min_Dh*Dh_main))**(1./(n_grids-1))

        Sx_list = [Sx_target]
        Sy_list = [Sy_target]
        Dh_list = [Dh_target]



        for i_grid in xrange(1,n_grids):
            Sx_list.append(Sx_list[-1]/f_exact)
            Sy_list.append(Sy_list[-1]/f_exact)
            Dh_list.append(Dh_list[-1]/f_exact)

            
        Sx_list = Sx_list[::-1]
        Sy_list = Sy_list[::-1]
        Dh_list = Dh_list[::-1] 
        pic_list = [pic_main]



        for i_grid in xrange(n_grids):
            x_min_int_curr = -Sx_list[i_grid]/2 + x_center_target
            x_max_int_curr = Sx_list[i_grid]/2 + x_center_target
            y_min_int_curr = -Sy_list[i_grid]/2 + y_center_target
            y_max_int_curr = Sy_list[i_grid]/2 + y_center_target
            Dh_int_curr = Dh_list[i_grid]
            print 'GRID %d/%d'%(i_grid,n_grids)
            pic_list.append(AddInternalGrid(pic_list[-1], x_min_int_curr, x_max_int_curr, y_min_int_curr, 
                                y_max_int_curr, Dh_int_curr, N_nodes_discard))
           
                                
        pic_list = pic_list[1:]
        self.n_grids = n_grids
        self.f_exact = f_exact
        self.pic_list = pic_list
        self.target = target
        self.f_telescope = f_telescope                        
                         
        self.scatter = self.pic_list[-1].scatter
        self.solve = self.pic_list[-1].solve
        self.gather = self.pic_list[-1].gather
        self.gather_phi = self.pic_list[-1].gather_phi
