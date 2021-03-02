import numpy as np

from .base import FieldMap
from . import linear_interpolators as li

class TriLinearInterpolatedFieldMap(FieldMap):

    def __init__(self, rho=None, phi=None,
                 x_grid=None, y_grid=None, z_grid=None,
                 dx=None, dy=None, dz=None,
                 nx=None, ny=None, nz=None,
                 x_range=None, y_range=None, z_range=None,
                 solver=None,
                 updatable=True,
                 context=None):

        self.updatable = updatable

        self._x_grid = _configure_grid('x', x_grid, dx, x_range, nx)
        self._y_grid = _configure_grid('y', y_grid, dy, y_range, ny)
        self._z_grid = _configure_grid('z', z_grid, dz, z_range, nz)

        # Prepare arrays
        self._rho = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64, order='F')
        self._phi = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64, order='F')
        self._dphi_dx = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64, order='F')
        self._dphi_dy = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64, order='F')
        self._dphi_dz = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64, order='F')

        # Set phi
        self.update_rho(rho, force=True)

        # Set phi
        self.update_phi(phi, force=True)

    @property
    def x_grid(self):
        return self._x_grid

    @property
    def y_grid(self):
        return self._y_grid

    @property
    def z_grid(self):
        return self._z_grid

    @property
    def nx(self):
        return len(self.x_grid)

    @property
    def ny(self):
        return len(self.y_grid)

    @property
    def nz(self):
        return len(self.z_grid)

    @property
    def dx(self):
        return self.x_grid[1] - self.x_grid[0]

    @property
    def dy(self):
        return self.y_grid[1] - self.y_grid[0]

    @property
    def dz(self):
        return self.z_grid[1] - self.z_grid[0]

    def get_values_at_points(self,
            x, y, z=0,
            return_rho=False,
            return_phi=False,
            return_dphi_dx=False,
            return_dphi_dy=False,
            return_dphi_dz=False):
        pass
        raise ValueError('To be implemented!')

    def update_rho(self, rho, reset=True, force=False):

        if not force:
            self._assert_updatable()

        if reset:
            self._rho[:,:,:] = rho
        else:
            raise ValueError('Not implemented!')

    def update_phi(self, phi, reset=True, force=False):

        if not force:
            self._assert_updatable()

        if reset:
            self._phi[:,:,:] = phi
        else:
            raise ValueError('Not implemented!')

        # Compute gradient
        self._dphi_dx[1:self.nx-1,:,:] = 1/(2*self.dx)*(self._phi[2:,:,:]-self._phi[:-2,:,:])
        self._dphi_dy[:,1:self.ny-1,:] = 1/(2*self.dy)*(self._phi[:,2:,:]-self._phi[:,:-2,:])
        self._dphi_dz[:,:,1:self.nz-1] = 1/(2*self.dz)*(self._phi[:,:,2:]-self._phi[:,:,:-2])

    def update_phi_from_rho(self, solver=None):

        raise ValueError('To be implemented!')
        self._assert_updatable()

        if solver is None:
            if hasattr(self, 'solver'):
                solver = self.solver
            else:
                raise ValueError('I have no solver to compute phi!')

    def update_rho_from_particles(x_p, y_p, z_p, ncharges_p, q0, reset=True):
        '''
        If reset is false charge density is added to the stored one
        '''

        raise ValueError('To be implemented!')
        self._assert_updatable()

    def update_all_from_particles(x_p, y_p, z_p, ncharges_p, q0, reset=True,
                                  solver=None):

        raise ValueError('To be implemented!')
        self._assert_updatable()

        self.update_rho_from_particles(
            x_p, y_p, z_p, ncharges_p, q0, reset=reset)

        self.update_phi_from_rho(solver=solver)

    def generate_solver(self, solver_type):
        raise ValueError('To be implemented!')
        return solver


def _configure_grid(vname, v_grid, dv, v_range, nv):

    # Check input consistency
    if v_grid is not None:
        assert dv is None, (f'd{vname} cannot be given '
                            f'if {vname}_grid is provided ')
        assert nv is None, (f'n{vname} cannot be given '
                            f'if {vname}_grid is provided ')
        assert v_range is None, (f'{vname}_range cannot be given '
                                 f'if {vname}_grid is provided')
        ddd = np.diff(v_grid)
        assert np.allclose(ddd,ddd[0]), (f'{vname}_grid must be '
                                          'unifirmly spaced')
    else:
        assert v_range is not None, (f'{vname}_grid or {vname}_range '
                                     f'must be provided')
        assert len(v_range)==2, (f'{vname}_range must be in the form '
                                 f'({vname}_min, {vname}_max)')
        if dv is not None:
            assert nv is None, (f'n{vname} cannot be given '
                                    f'if d{vname} is provided ')
            v_grid = np.arange(v_range[0], v_range[1]+0.1*dv, dv)
        else:
            assert nv is not None, (f'n{vname} must be given '
                                    f'if d{vname} is not provided ')
            v_grid = np.linspace(v_range[0], v_range[1], nv)

    return v_grid

# ## First sketch ##
#
# class InterpolatedFieldMap(FieldMap):
# 
#     def __init__(self, rho=None, phi=None,
#                  x_grid=None, y_grid=None, z_grid=None,
#                  dx=None, dy=None, dz=None,
#                  x_range=None, y_range=None, z_range=None,
#                  xy_interp_method='linear',
#                  z_interp_method='linear',
#                  context=None):
#         '''
#         interp_methods can be 'linear' or 'cubic'
#         '''
# 
#         # 1D, 2D or 3D is inferred from the matrix size 
#         pass
# 
#     def get_values_at_points(self,
#             x, y, z=0,
#             return_rho=False,
#             return_phi=False,
#             return_dphi_dx=False,
#             return_dphi_dy=False,
#             return_dphi_dz=False):
#         pass
# 
# 
# class InterpolatedFieldMapWithBoundary(FieldMap):
# 
#     def __init__(self, rho=None, phi=None,
#                  x_grid=None, y_grid=None, z_grid=None,
#                  dx=None, dy=None, dz=None,
#                  xy_interp_method='linear',
#                  z_interp_method='linear',
#                  boundary=None,
#                  context=None):
#         '''
#         Does the Shortley-Weller interpolation close to the boundary.
#         Might need to force 2D and linear for now.
#         '''
# 
#         pass
# 
#     def get_values_at_points(self,
#             x, y, z=0,
#             return_rho=False,
#             return_phi=False,
#             return_dphi_dx=False,
#             return_dphi_dy=False,
#             return_dphi_dz=False):
#         pass
