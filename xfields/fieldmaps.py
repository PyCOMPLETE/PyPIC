from abc import ABC, abstractmethod


class FieldMap(ABC):

    @abstractmethod
    def __init__(self, context=None, solver=None, solver_type=None, 
                 updatable=True, **kwargs):

        '''
        The init will have context argument, specipying the context
        in which wee store the state of the field map
        '''
        if solver is not None:
            'Check conmpatibility with grid'
            self.solver = solver
        elif solver=='generate':
            self.generate_solver(solver_type)

    @abstractmethod
    def generate_solver(self, solver_type):
        pass
        return solver

    @abstractmethod
    def get_data_and_singleparticle_code(self):
        '''
        To be defined, to inject element in 
        single-particle tracking
        '''
        pass

    @abstractmethod
    def get_values_at_points(self,
            x, y, z=0, 
            return_rho=False, 
            return_phi=False,
            return_dphi_dx=False, 
            return_dphi_dy=False, 
            return_dphi_dz=False):
        pass

    def update_rho(self, rho, reset=True):
        
        self._assert_updatable()
        
        self.rho = rho.copy()

    def update_phi(self, phi, reset=True):
        
        self._assert_updatable()

        self.phi = phi.copy()

    @abstractmethod
    def update_rho_from_particles(x_p, y_p, z_p, ncharges_p, q0, reset=True):
        '''
        If reset is false charge density is added to the stored one
        '''
        
        self._assert_updatable()

    def update_phi_from_rho(self, solver=None):
        
        self._assert_updatable()

        if solver is None:
            if hasattr(self, 'solver'):
                solver = self.solver
            else:
                raise ValueError('I have no solver to compute phi!')

    def update_all_from_particles(x_p, y_p, z_p, ncharges_p, q0, reset=True,
                                  solver=None):
        
        self._assert_updatable()

        self.update_rho_from_particles(
            x_p, y_p, z_p, ncharges_p, q0, reset=reset)

        self.update_phi_from_rho(solver=solver)

    def _assert_updatable(self):
        assert self.updatable, 'This FieldMap is not updatable!'


class BiGaussianFieldMap(FieldMap):
    '''
    Bassetti-Erskine
    Must be 2D, no closed form dor 3D in general...
    '''
    def __init__(self, charge, sigma_x, sigma_y, theta=0, 
                 context=None):
        'theta is a rotation angle, e.g. to handle coupling'

        # For now:
        assert (theta == 0), 'Rotation (theta != 0 not yet implemented)'

        pass

    def get_values_at_points(self,
            x, y, z=0, 
            return_rho=False, 
            return_phi=False,
            return_dphi_dx=False, 
            return_dphi_dy=False, 
            return_dphi_dz=False):


        '''
        To have the same behavior as for the others we might keep different 
        sigmas for rho and phi
        '''
        pass 

    def update_rho(self, rho, reset):
        raise ValueError('rho cannot be directly updated'
                         'for UpdatableBiGaussianFieldMap')

    def update_rho_from_particles(x_p, y_p, z_p, ncharges_p, q0, reset=True):

        assert reset, ('rho cannot be added (reset must be True) for '
                      'for UpdatableBiGaussianFieldMap')
        # Basically updates sigma_rhos

    def update_phi_from_rho(self, solver=None):
        
        assert (solver is None), ('no solver can be passed for'
                                  'UpdatableBiGaussianFieldMap')
        # Updates sigma_phi from sigma_rho
        pass  


class InterpolatedFieldMap(FieldMap): 

    def __init__(self, rho=None, phi=None, 
                 x_grid=None, y_grid=None, z_grid=None
                 dx=None, dy=None, dz=None,
                 x_range=None, y_range=None, z_range=None, 
                 xy_interp_method='linear',
                 z_interp_method='linear',
                 context=None):
        '''
        interp_methods can be 'linear' or 'cubic'
        '''

        # 1D, 2D or 3D is inferred from the matrix size 
        pass

    def get_values_at_points(self,
            x, y, z=0, 
            return_rho=False, 
            return_phi=False,
            return_dphi_dx=False, 
            return_dphi_dy=False, 
            return_dphi_dz=False):
        pass


class InterpolatedFieldMapWithBoundary(FieldMap): 

    def __init__(self, rho=None, phi=None, 
                 x_grid=None, y_grid=None, z_grid=None
                 dx=None, dy=None, dz=None,
                 xy_interp_method='linear',
                 z_interp_method='linear',
                 boundary=None
                 context=None):
        '''
        Does the Shortley-Weller interpolation close to the boundary.
        Might need to force 2D and linear for now.
        '''

        pass

    def get_values_at_points(self,
            x, y, z=0, 
            return_rho=False, 
            return_phi=False,
            return_dphi_dx=False, 
            return_dphi_dy=False, 
            return_dphi_dz=False):
        pass


class DualGridFieldMap(FieldMap):
    def __init__(self, external_grid, internal_grid_properties,
                 context=None):
        pass




 