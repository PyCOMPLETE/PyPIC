


class SpaceCharge2D(object):

    def __init__(self, 
                 update_on_track=False, # Decides if frozen or soft-gaussian
                 apply_z_kick=True
                 transverse_field_map=None, 
                 longitudinal_profile=None,
                 context=None,
                 ):
        pass

class SpaceCharge2DBiGaussian(SpaceChargeModulated2D):

    def __init__(self, 
                 update_on_track=False, # Decides if frozen or soft-gaussian
                 apply_z_kick=True
                 sigma_x=None, sigma_y=None,
                 longitudinal_mode='Gaussian', 
                 sigma_z=None, 
                 z_grid=None, dz=None,
                 z_interp_method='linear',
                 context=None,
                 ):
        pass

class SpaceCharge2DInterpMap(SpaceChargeModulated2D):

    def __init__(self, 
                 update_on_track=False, # Decides if frozen or kick
                 apply_z_kick=True
                 rho=None, phi=None, 
                 x_grid=None, y_grid=None,
                 dx=None, dy=None,
                 x_range=None, y_range=None,
                 xy_interp_method='linear',
                 longitudinal_mode='Gaussian', 
                 sigma_z=None, 
                 z_grid=None, dz=None,
                 z_interp_method='linear',
                 contex=None,
                 ):
        pass

class SpaceCharge3D(object):

    def __init__(self, 
                 update_on_track=False, # Decides if frozen or kick
                 mode='2.5D',
                 apply_z_kick=True, 
                 rho=None, phi=None, 
                 x_grid=None, y_grid=None, z_grid=None
                 dx=None, dy=None, dz=None,
                 x_range=None, y_range=None, z_range=None, 
                 xy_interp_method='linear',
                 z_interp_method='linear',
                 context=None,
                 )
        '''
        Needed when transverse normalized distribution changes along z.

        mode can be 2.5D or 3D
        '''


