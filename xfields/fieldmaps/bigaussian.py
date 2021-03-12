from .base import FieldMap

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

