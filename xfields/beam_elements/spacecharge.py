from xfields.platforms import XfCpuPlatform
from xfields import TriLinearInterpolatedFieldMap

class SpaceCharge3D(object):

    def __init__(self,
                 length=None,
                 update_on_track=False, # Decides if frozen or PIC
                 apply_z_kick=True,
                 rho=None, phi=None,
                 x_grid=None, y_grid=None, z_grid=None,
                 dx=None, dy=None, dz=None,
                 nx=None, ny=None, nz=None,
                 x_range=None, y_range=None, z_range=None,
                 solver=None,
                 platform=XfCpuPlatform()):
        '''
        Needed when transverse normalized distribution changes along z.
        mode can be 2.5D or 3D
        '''

        self.length = length
        self.update_on_track = update_on_track
        self.apply_z_kick = apply_z_kick

        fieldmap = TriLinearInterpolatedFieldMap(
                        rho=rho, phi=phi,
                        x_grid=z_grid, y_grid=y_grid, z_grid=z_grid,
                        x_range=z_range, y_range=y_range, z_range=z_range,
                        dx=dx, dy=dy, dz=dz,
                        nx=nx, ny=ny, nz=nz,
                        solver=solver,
                        updatable=update_on_track,
                        platform=platform)

        self.fieldmap = fieldmap



class SpaceCharge2D(object):

    def __init__(self,
                 update_on_track=False, # Decides if frozen or soft-gaussian
                 apply_z_kick=True,
                 transverse_field_map=None,
                 longitudinal_profile=None,
                 platform=None,
                 ):
        pass

class SpaceCharge2DBiGaussian(SpaceCharge2D):

    def __init__(self,
                 update_on_track=False, # Decides if frozen or soft-gaussian
                 apply_z_kick=True,
                 sigma_x=None, sigma_y=None,
                 longitudinal_mode='Gaussian',
                 sigma_z=None,
                 z_grid=None, dz=None,
                 z_interp_method='linear',
                 platform=None,
                 ):
        pass

class SpaceCharge2DInterpMap(SpaceCharge2D):

    def __init__(self,
                 update_on_track=False, # Decides if frozen or kick
                 apply_z_kick=True,
                 rho=None, phi=None,
                 x_grid=None, y_grid=None,
                 dx=None, dy=None,
                 x_range=None, y_range=None,
                 xy_interp_method='linear',
                 longitudinal_mode='Gaussian',
                 sigma_z=None,
                 z_grid=None, dz=None,
                 z_interp_method='linear',
                 context=None,
                 ):
        pass


