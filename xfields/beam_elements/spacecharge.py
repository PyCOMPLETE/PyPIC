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
                 gamma0=None,
                 platform=XfCpuPlatform()):
        '''
        Needed when transverse normalized distribution changes along z.
        mode can be 2.5D or 3D
        '''

        self.length = length
        self.update_on_track = update_on_track
        self.apply_z_kick = apply_z_kick

        if solver=='FFTSolver3D':
            assert gamma0 is not None, ('To use FFTSolver3D '
                                        'gamma0 must be provided')

        if gamma0 is not None:
            scale_coordinates_in_solver=(1.,1., float(gamma0))
        else:
            scale_coordinates_in_solver=(1.,1.,1.)

        fieldmap = TriLinearInterpolatedFieldMap(
                    rho=rho, phi=phi,
                    x_grid=z_grid, y_grid=y_grid, z_grid=z_grid,
                    x_range=x_range, y_range=y_range, z_range=z_range,
                    dx=dx, dy=dy, dz=dz,
                    nx=nx, ny=ny, nz=nz,
                    solver=solver,
                    scale_coordinates_in_solver=scale_coordinates_in_solver,
                    updatable=update_on_track,
                    platform=platform)

        self.fieldmap = fieldmap

    def track(self, particles):

        if self.update_on_track:
            self.fieldmap.update_from_particles(
                    x_p=particles.x,
                    y_p=particles.y,
                    z_p=particles.zeta,
                    ncharges_p=particles.weight,
                    q0=particles.q0*particles.echarge)


        res = self.fieldmap.get_values_at_points(
                            x=particles.x, y=particles.y, z=particles.zeta,
                            return_rho=False, return_phi=False,
                            return_dphi_dz=self.apply_z_kick)
        # res = [dphi_dx, dphi_dy, (dphi_z)]

        #Build factor
        beta0 = particles.beta0
        clight = float(particles.clight)
        charge_mass_ratio = (particles.chi*particles.echarge*particles.q0
                                /(particles.mass0*particles.echarge/(clight*clight)))
        gamma0 = particles.gamma0
        beta0 = particles.beta0
        factor = -(charge_mass_ratio*self.length*(1.-beta0*beta0)
                    /(gamma0*beta0*beta0*clight*clight))

        # Kick particles
        particles.px += factor*res[0]
        particles.py += factor*res[1]
        if self.apply_z_kick:
            particles.delta += factor*res[2]



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


