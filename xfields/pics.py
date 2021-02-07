from abc import ABC, abstractmethod
from .fieldmaps import (FieldMap, BiGaussianFieldMap, 
    InterpolatedFieldMap, DualGridFieldMap)


class UpdatableFieldMap(FieldMap):

    def __init__(self, solver=None, solver_type=None, **kwargs):

        super().__init__(**kwargs)

        if solver is not None:
            'Check conmpatibility with grid'
            self.solver = solver
        elif solver=='generate':
            self.generate_solver(solver_type)

    
    def update_rho(self, rho, reset=True):
        self.rho = rho.copy()

    def update_phi(self, phi, reset=True):
        self.phi = phi.copy()

    @abstractmethod
    def update_rho_from_particles(x_p, y_p, z_p, ncharges_p, q0, reset=True):
        '''
        If reset is false charge density is added to the stored one
        '''

    def update_phi_from_rho(self, solver=None):

        if solver is None:
            if hasattr(self, 'solver'):
                solver = self.solver
            else:
                raise ValueError('I have no solver to compute phi!')

    def update_all_from_particles(x_p, y_p, z_p, ncharges_p, q0, reset=True,
                                  solver=None):
        
        self.update_rho_from_particles(
            x_p, y_p, z_p, ncharges_p, q0, reset=reset)

        self.update_phi_from_rho(solver=solver)

    @abstractmethod
    def generate_solver(self, solver_type):
        pass


class UpdatableBiGaussianFieldMap(BiGaussianFieldMap, UpdatableFieldMap):

    '''
    To have the same behavior as for the others we might keep different 
    sigmas for rho and phi
    '''

    def update_rho(self, rho, reset):
        raise ValueError('rho cannot be directly updated'
                         'for UpdatableBiGaussianFieldMap')

    def update_rho_from_particles(x_p, y_p, z_p, ncharges_p, q0, reset=True):

        assert reset, ('rho cannot be added for '
                      'for UpdatableBiGaussianFieldMap')
        # Basically updates sigma_rhos

    def update_phi_from_rho(self, solver=None):
        
        assert (solver is None), ('no solver can be passed for'
                                  'UpdatableBiGaussianFieldMap')
        # Updates sigma_phi from sigma_rho
        pass


class UpdatableInterpolatedFieldMap(InterpolatedFieldMap, UpdatableFieldMap):
    pass
    

class UpdatableDualGridFieldMap(InterpolatedFieldMap):
    def __init__(self, external_grid, internal_grid_properties,
                 context=None):
        pass





