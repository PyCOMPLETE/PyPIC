import numpy as np
from . import FiniteDifferences_Staircase_SquareGrid as PIC_FD
from . import FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
from . import simple_polygon as spoly
from .PyPIC_Scatter_Gather import PyPIC_Scatter_Gather
from scipy.constants import e, epsilon_0

qe = e
eps0 = epsilon_0
class AddInternalGrid(PyPIC_Scatter_Gather):
    def __init__(self, pic_external, x_min_internal, x_max_internal, y_min_internal, y_max_internal, Dh_internal, N_nodes_discard,
                sparse_solver = 'PyKLU', include_solver = True):
        

        #build boundary for refinement grid
        box_internal = spoly.SimplePolygon({'Vx':np.array([x_max_internal, x_min_internal, x_min_internal, x_max_internal]),
                                'Vy':np.array([y_max_internal, y_max_internal, y_min_internal, y_min_internal])})
        if include_solver:
            self.pic_internal = PIC_FD.FiniteDifferences_Staircase_SquareGrid(chamb = box_internal, Dh = Dh_internal, 
                                    remove_external_nodes_from_mat=False, sparse_solver=sparse_solver, include_solver = True)
            #check if the internal grid lies inside the chamber
            x_border = self.pic_internal.xn[self.pic_internal.flag_border_n]
            y_border = self.pic_internal.yn[self.pic_internal.flag_border_n]
            if pic_external.chamb.is_outside(x_border, y_border).any() == True:
                raise ValueError('The internal grid is outside the chamber!')

            

        else:
            self.pic_internal = PIC_FD.FiniteDifferences_Staircase_SquareGrid(chamb = box_internal, Dh = Dh_internal, 
                                    remove_external_nodes_from_mat=False, sparse_solver=sparse_solver, include_solver = False)
                                    
                                    
        self.sparse_solver = sparse_solver
        self.pic_external = pic_external
        self.chamb = self.pic_external.chamb	
        self.x_min_internal = x_min_internal
        self.x_max_internal = x_max_internal
        self.y_min_internal = y_min_internal
        self.y_max_internal = y_max_internal
        self.Dh_internal = Dh_internal
        self.N_nodes_discard = N_nodes_discard
        self.D_discard = N_nodes_discard*Dh_internal	

    def scatter(self, x_mp, y_mp, nel_mp, charge = -qe, flag_add=False):
        self.pic_external.scatter(x_mp, y_mp, nel_mp, charge, flag_add)
        self.pic_internal.scatter(x_mp, y_mp, nel_mp, charge, flag_add)

         
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
        
    def gather_rho(self, x_mp, y_mp):
        mask_internal = np.logical_and(\
            np.logical_and(x_mp > self.x_min_internal + self.D_discard, 
                           x_mp < self.x_max_internal - self.D_discard),
            np.logical_and(y_mp > self.y_min_internal + self.D_discard, 
                           y_mp < self.y_max_internal - self.D_discard))
                           
        mask_external = np.logical_not(mask_internal)
        
        rho_sc_n_external = self.pic_external.gather_rho(x_mp[mask_external], y_mp[mask_external])
        rho_sc_n_internal = self.pic_internal.gather_rho(x_mp[mask_internal], y_mp[mask_internal])
        
        rho_sc_n = 0.*x_mp
        
        rho_sc_n[mask_external] = rho_sc_n_external
        rho_sc_n[mask_internal] = rho_sc_n_internal

        return rho_sc_n

    def solve(self, rho = None, flag_verbose = False):
        if rho is not None:
            raise ValueError('rho matrix cannot be provided in multigrid mode!')
        self.pic_external.solve(flag_verbose = flag_verbose)
        self.pic_internal.solve(flag_verbose = flag_verbose, pic_external=self.pic_external)
    

    def get_state_object(self):
        state_external = self.pic_external.get_state_object()
        
        state = AddInternalGrid(state_external, self.x_min_internal, self.x_max_internal, self.y_min_internal, 
                self.y_max_internal, self.Dh_internal, self.N_nodes_discard, sparse_solver=self.sparse_solver, include_solver = False)
                
        state.pic_internal.rho = self.pic_internal.rho.copy()
        state.pic_internal.phi = self.pic_internal.phi.copy()				
        state.pic_internal.efx = self.pic_internal.efx.copy()	
        state.pic_internal.efy = self.pic_internal.efy.copy()
        
        return state	
                            
    def solve_states(self, states):
        states = np.atleast_1d(states)
        states_external = [state.pic_external for state in states]
        states_internal = [state.pic_internal for state in states]
        self.pic_external.solve_states(states_external)
        self.pic_internal.solve_states(states_internal, pic_s_external=states_external)
    
    @property
    def rho(self):
        return self.pic_internal.rho	
        
    @property
    def phi(self):
        return self.pic_internal.phi
        
    @property
    def efx(self):
        return self.pic_internal.efx
        
    @property
    def efy(self):
        return self.pic_internal.efy
        
    
        
class AddMultiGrids(PyPIC_Scatter_Gather):
    def __init__(self, pic_main, grids, sparse_solver='PyKLU', include_solver = True):


        n_grids = len(grids)
        pic_list = [pic_main]
        for ii in range(n_grids):
            print('GRID %d/%d'%(ii,n_grids))
            
            x_min_internal = grids[ii]['x_min_internal']
            x_max_internal = grids[ii]['x_max_internal']
            y_min_internal = grids[ii]['y_min_internal']
            y_max_internal = grids[ii]['y_max_internal']
            Dh_internal = grids[ii]['Dh_internal']
            N_nodes_discard = grids[ii]['N_nodes_discard']

            pic_list.append(AddInternalGrid(pic_list[-1], x_min_internal, x_max_internal, y_min_internal, 
                                y_max_internal, Dh_internal, N_nodes_discard, sparse_solver=sparse_solver, 
                                include_solver = include_solver))

                            
        pic_list = pic_list[1:]
        self.n_grids = n_grids
        self.pic_list = pic_list
        self.pic_main = pic_main
        self.grids = grids			 
        
                         
        self.scatter = self.pic_list[-1].scatter
        self.solve = self.pic_list[-1].solve
        self.gather = self.pic_list[-1].gather
        self.gather_phi = self.pic_list[-1].gather_phi
        self.gather_rho = self.pic_list[-1].gather_rho
        
        self.Dh = self.pic_list[-1].pic_internal.Dh
        self.xg = self.pic_list[-1].pic_internal.xg
        self.Nxg = self.pic_list[-1].pic_internal.Nxg
        self.bias_x = self.pic_list[-1].pic_internal.bias_x
        self.yg = self.pic_list[-1].pic_internal.yg
        self.Nyg = self.pic_list[-1].pic_internal.Nyg
        self.bias_y = self.pic_list[-1].pic_internal.bias_y
        
        self.solve_states = self.pic_list[-1].solve_states
        self.get_state_object = self.pic_list[-1].get_state_object

        
    @property
    def rho(self):
        return self.pic_list[-1].pic_internal.rho	
        
    @property
    def phi(self):
        return self.pic_list[-1].pic_internal.phi
        
    @property
    def efx(self):
        return self.pic_list[-1].pic_internal.efx
        
    @property
    def efy(self):
        return self.pic_list[-1].pic_internal.efy
        
        
        
        

class AddTelescopicGrids(AddMultiGrids):
    def __init__(self, pic_main, f_telescope, target_grid, N_nodes_discard, N_min_Dh_main,
                    sparse_solver='PyKLU'):
        
        x_min_target = target_grid['x_min_target']
        x_max_target = target_grid['x_max_target']
        y_min_target = target_grid['y_min_target']
        y_max_target = target_grid['y_max_target']
        Dh_target = target_grid['Dh_target']
        
        Dh_main = pic_main.Dh

        
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

        if S_target >= (N_min_Dh_main*Dh_main):
            n_grids = 1
        else:
            n_grids = int(np.ceil(np.log(S_target/(N_min_Dh_main*Dh_main))/np.log(f_telescope)))+1
        
        print('%d grids needed'%n_grids)
        
        if n_grids == 1:
            f_exact = None #it's not used
        else:
            f_exact = (S_target/(N_min_Dh_main*Dh_main))**(1./(n_grids-1))

        Sx_list = [Sx_target]
        Sy_list = [Sy_target]
        Dh_list = [Dh_target]



        for i_grid in range(1,n_grids):
            Sx_list.append(Sx_list[-1]/f_exact)
            Sy_list.append(Sy_list[-1]/f_exact)
            Dh_list.append(Dh_list[-1]/f_exact)

            
        Sx_list = Sx_list[::-1]
        Sy_list = Sy_list[::-1]
        Dh_list = Dh_list[::-1] 
        pic_list = [pic_main]


        grids = []
        for i_grid in range(n_grids):
            x_min_int_curr = -Sx_list[i_grid]/2 + x_center_target
            x_max_int_curr = Sx_list[i_grid]/2 + x_center_target
            y_min_int_curr = -Sy_list[i_grid]/2 + y_center_target
            y_max_int_curr = Sy_list[i_grid]/2 + y_center_target
            Dh_int_curr = Dh_list[i_grid]

            grids.append({\
            'x_min_internal':x_min_int_curr,
            'x_max_internal':x_max_int_curr,
            'y_min_internal':y_min_int_curr,
            'y_max_internal':y_max_int_curr,
            'Dh_internal':Dh_int_curr,
            'N_nodes_discard':N_nodes_discard})
            
        self.target_grid = target_grid
        self.f_telescope = f_telescope   
        self.f_exact = f_exact
        self.N_nodes_discard = N_nodes_discard
        self.N_min_Dh_main = N_min_Dh_main
        
        super(AddTelescopicGrids, self).__init__(pic_main=pic_main, grids=grids, sparse_solver=sparse_solver)


