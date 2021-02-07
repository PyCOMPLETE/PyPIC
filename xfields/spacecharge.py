


class SpaceChargeModulated2D(object):

    def __init__(self, 
                 update_on_track=False, # Decides if frozen or soft-gaussian
                 apply_z_kick=True
                 transverse_field_map=None, 
                 longitudinal_profile=None,
                 contex=None,
                 )

class SpaceChargeModulated2DBiGaussian(SpaceChargeModulated2D):

    def __init__(self, 
                 update_on_track=False, # Decides if frozen or soft-gaussian
                 apply_z_kick=True
                 sigma_x=None, sigma_y=None,
                 longitudinal_mode='Gaussian', 
                 sigma_z=None, 
                 z_grid=None, dz=None,
                 z_interp_method='linear',
                 contex=None,
                 )

class SpaceChargeModulated2DInterpMap(SpaceChargeModulated2D):

    def __init__(self, 
                 update_on_track=False, # Decides if frozen or kick
                 apply_z_kick=True
                 sigma_x=None, sigma_y=None,
                 longitudinal_mode='Gaussian', 
                 sigma_z=None, 
                 z_grid=None, dz=None,
                 z_interp_method='linear',
                 contex=None,
                 )

