from abc import ABC, abstractmethod


class FieldMap(ABC):

    '''
    The init will have context argument, specipying the context
    in which wee store the state of the field map
    '''

    @abstractmethod
    def get_values_at_points(self,
            x, y, z=0, 
            return_rho=False, 
            return_phi=False,
            return_dphi_dx=False, 
            return_dphi_dy=False, 
            return_dphi_dz=False):
        pass

    @abstractmethod
    def get_data_and_singleparticle_code(self):
        '''
        To be defined, to inject element in 
        single-particle tracking
        '''


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
        pass   


class InterpolatedFieldMap(FieldMap): 

    def __init__(self, rho=None, phi=None, 
                 x_grid=None, y_grid=None, z_grid=None
                 dx=None, dy=None, dz=None,
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


class DualGridFieldMap(FieldMap):
    def __init__(self, external_grid, internal_grid_properties,
                 context=None):
        pass




 