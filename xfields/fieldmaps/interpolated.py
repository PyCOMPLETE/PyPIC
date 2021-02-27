from .base import FieldMap

class TriLinearInterpolatedFieldMap(FieldMap):

    def __init__(self, rho=None, phi=None,
                 x_grid=None, y_grid=None, z_grid=None,
                 dx=None, dy=None, dz=None,
                 x_range=None, y_range=None, z_range=None,
                 solver=None, solver_type=None,
                 updatable=True,
                 context=None):
        pass


    def _set_grid(self, vname, v_grid, dv, v_range, nv):

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

            setattr(self, f'{vname}_grid', v_grid)


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
