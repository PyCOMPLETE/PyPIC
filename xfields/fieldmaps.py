from abc import ABC, abstractmethod


class FieldMap(ABS):

    @abstractmethod
    def get_values_at_points(
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
                 z_interp_method='linear'):
        '''
        interp_methods can be 'linear' or 'cubic'
        '''

        # 1D, 2D or 3D is inferred from the matrix size 
        pass

    def get_values_at_points(
            x, y, z=0, 
            return_rho=False, 
            return_phi=False,
            return_dphi_dx=False, 
            return_dphi_dy=False, 
            return_dphi_dz=False):
        pass


class BiGaussianFieldMap(FieldMap):
    '''
    Bassetti-Erskine
    Must be 2D, no closed form dor 3D in general...
    '''
    def __init__(self, charge, sigma_x, sigma_y):
        pass

    def get_values_at_points(
            x, y, z=0, 
            return_rho=False, 
            return_phi=False,
            return_dphi_dx=False, 
            return_dphi_dy=False, 
            return_dphi_dz=False):
        pass    