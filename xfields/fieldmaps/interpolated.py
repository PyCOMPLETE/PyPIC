import numpy as np

from .base import FieldMap
from ..solvers.fftsolvers import FFTSolver3D
from ..platforms import XfCpuPlatform

class TriLinearInterpolatedFieldMap(FieldMap):

    def __init__(self, rho=None, phi=None,
                 x_grid=None, y_grid=None, z_grid=None,
                 dx=None, dy=None, dz=None,
                 nx=None, ny=None, nz=None,
                 x_range=None, y_range=None, z_range=None,
                 solver=None,
                 updatable=True,
                 platform=XfCpuPlatform()):

        self.updatable = updatable

        self.platform = platform

        self._x_grid = _configure_grid('x', x_grid, dx, x_range, nx)
        self._y_grid = _configure_grid('y', y_grid, dy, y_range, ny)
        self._z_grid = _configure_grid('z', z_grid, dz, z_range, nz)

        # Prepare arrays (contiguous to use a single pointer in C/GPU)
        self._maps_buffer_dev = platform.nparray_to_platform_mem(
                np.zeros((self.nx, self.ny, self.nz, 5),
                         dtype=np.float64, order='F'))

        # These are slices (they are are on the platform)
        self._rho_dev = self._maps_buffer_dev[:, :, :, 0]
        self._phi_dev = self._maps_buffer_dev[:, :, :, 1]
        self._dphi_dx_dev = self._maps_buffer_dev[:, :, :, 2]
        self._dphi_dy_dev = self._maps_buffer_dev[:, :, :, 3]
        self._dphi_dz_dev = self._maps_buffer_dev[:, :, :, 4]


        if isinstance(solver, str):
            self.solver = self.generate_solver(solver)
        else:
            self.solver = solver

        # Set rho
        if rho is not None:
            self.update_rho(rho, force=True)

        # Set phi
        if phi is not None:
            self.update_phi(phi, force=True)
        else:
            if solver is not None:
                self.update_phi_from_rho()

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

    @property
    def rho(self):
        return self._rho_dev

    @property
    def phi(self):
        return self._phi_dev

    def get_values_at_points(self,
            x, y, z,
            return_rho=True,
            return_phi=True,
            return_dphi_dx=True,
            return_dphi_dy=True,
            return_dphi_dz=True):

        assert len(x) == len(y) == len(z)

        pos_in_buffer_of_maps_to_interp = []
        mapsize = self.nx*self.ny*self.nz
        if return_rho:
            pos_in_buffer_of_maps_to_interp.append(0*mapsize)
        if return_phi:
            pos_in_buffer_of_maps_to_interp.append(1*mapsize)
        if return_dphi_dx:
            pos_in_buffer_of_maps_to_interp.append(2*mapsize)
        if return_dphi_dy:
            pos_in_buffer_of_maps_to_interp.append(3*mapsize)
        if return_dphi_dz:
            pos_in_buffer_of_maps_to_interp.append(4*mapsize)

        pos_in_buffer_of_maps_to_interp = self.platform.nparray_to_platform_mem(
                        np.array(pos_in_buffer_of_maps_to_interp, dtype=np.int32))
        nmaps_to_interp = len(pos_in_buffer_of_maps_to_interp)
        buffer_out = self.platform.nparray_to_platform_mem(
                np.zeros(nmaps_to_interp * len(x), dtype=np.float64))
        if nmaps_to_interp > 0:
            self.platform.kernels.m2p_rectmesh3d(
                    nparticles=len(x),
                    x=x, y=y, z=z,
                    x0=self.x_grid[0], y0=self.y_grid[0], z0=self.z_grid[0],
                    dx=self.dx, dy=self.dy, dz=self.dz,
                    nx=self.nx, ny=self.ny, nz=self.nz,
                    n_quantities=nmaps_to_interp,
                    offsets_mesh_quantities=pos_in_buffer_of_maps_to_interp,
                    mesh_quantity=self._maps_buffer_dev,
                    particles_quantity=buffer_out)

        # Split buffer 
        particles_quantities = [buffer_out[ii*len(x):(ii+1)*len(x)]
                                        for ii in range(nmaps_to_interp)]

        return particles_quantities

    def update_rho(self, rho, reset=True, force=False):

        if not force:
            self._assert_updatable()

        if reset:
            self._rho_dev[:,:,:] = rho
        else:
            raise ValueError('Not implemented!')

    def update_phi(self, phi, reset=True, force=False):

        if not force:
            self._assert_updatable()

        if reset:
            self._phi_dev[:,:,:] = phi
        else:
            raise ValueError('Not implemented!')

        # Compute gradient
        self._dphi_dx_dev[1:self.nx-1,:,:] = 1/(2*self.dx)*(
                self._phi_dev[2:,:,:]-self._phi_dev[:-2,:,:])
        self._dphi_dy_dev[:,1:self.ny-1,:] = 1/(2*self.dy)*(
                self._phi_dev[:,2:,:]-self._phi_dev[:,:-2,:])
        self._dphi_dz_dev[:,:,1:self.nz-1] = 1/(2*self.dz)*(
                self._phi_dev[:,:,2:]-self._phi_dev[:,:,:-2])

    def update_phi_from_rho(self, solver=None):

        self._assert_updatable()

        if solver is None:
            if hasattr(self, 'solver'):
                solver = self.solver
            else:
                raise ValueError('I have no solver to compute phi!')

        new_phi = solver.solve(self._rho_dev)
        self.update_phi(new_phi)


    def update_from_particles(self, x_p, y_p, z_p, ncharges_p, q0, reset=True,
                            update_phi=True, solver=None, force=False):

        if not force:
            self._assert_updatable()

        if reset:
            self._rho_dev[:,:,:] = 0.

        assert len(x_p) == len(y_p) == len(z_p) == len(ncharges_p)

        self.platform.kernels.p2m_rectmesh3d(
                nparticles=len(x_p),
                x=x_p, y=y_p, z=z_p,
                part_weights=q0*ncharges_p,
                x0=self.x_grid[0], y0=self.y_grid[0], z0=self.z_grid[0],
                dx=self.dx, dy=self.dy, dz=self.dz,
                nx=self.nx, ny=self.ny, nz=self.nz,
                grid1d=self._rho_dev)

        if update_phi:
            self.update_phi_from_rho(solver=solver)

    def generate_solver(self, solver):

        if solver == 'FFTSolver3D':
            solver = FFTSolver3D(
                    dx=self.dx, dy=self.dy, dz=self.dz,
                    nx=self.nx, ny=self.ny, nz=self.nz)
        else:
            raise ValueError(f'solver name {solver} not recognized')

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
