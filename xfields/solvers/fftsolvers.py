import numpy as np
from scipy.constants import epsilon_0
from numpy import pi

from .base import Solver
from ..platforms import XfCpuPlatform

class FFTSolver2D(Solver):

    def solve(self, rho):
        pass

class FFTSolver3D(Solver):

    def __init__(self, dx, dy, dz, nx, ny, nz, platform=XfCpuPlatform()):

        # Prepare arrays
        workspace_dev = platform.nparray_to_platform_mem(
                    np.zeros((2*nx, 2*ny, 2*nz), dtype=np.complex128, order='F'))


        # Build grid for primitive function
        xg_F = np.arange(0, nx+2) * dx - dx/2
        yg_F = np.arange(0, ny+2) * dy - dy/2
        zg_F = np.arange(0, nz+2) * dz - dz/2
        XX_F, YY_F, ZZ_F = np.meshgrid(xg_F, yg_F, zg_F, indexing='ij')

        # Compute primitive
        F_temp = primitive_func_3d(XX_F, YY_F, ZZ_F)

        # Integrated Green Function (I will transform inplace)
        gint_rep= np.zeros((2*nx, 2*ny, 2*nz), dtype=np.complex128, order='F')
        gint_rep[:nx+1, :ny+1, :nz+1] = (F_temp[ 1:,  1:,  1:]
                                       - F_temp[:-1,  1:,  1:]
                                       - F_temp[ 1:, :-1,  1:]
                                       + F_temp[:-1, :-1,  1:]
                                       - F_temp[ 1:,  1:, :-1]
                                       + F_temp[:-1,  1:, :-1]
                                       + F_temp[ 1:, :-1, :-1]
                                       - F_temp[:-1, :-1, :-1])

        # Replicate
        # To define how to make the replicas I have a look at:
        # np.abs(np.fft.fftfreq(10))*10
        # = [0., 1., 2., 3., 4., 5., 4., 3., 2., 1.]
        gint_rep[nx+1:, :ny, :nz] = gint_rep[nx-1:0:-1, :ny, :nz]
        gint_rep[:nx, ny+1:, :nz] = gint_rep[:nx, ny-1:0:-1, :nz]
        gint_rep[nx+1:, ny+1:, :nz] = gint_rep[nx-1:0:-1, ny-1:0:-1, :nz]
        gint_rep[:nx, :ny, nz+1:] = gint_rep[:nx, :ny, nz-1:0:-1]
        gint_rep[nx+1:, :ny, nz+1:] = gint_rep[nx-1:0:-1,  :ny, nz-1:0:-1]
        gint_rep[:nx, ny+1:, nz+1:] = gint_rep[:nx, ny-1:0:-1, nz-1:0:-1]
        gint_rep[nx+1:, ny+1:, nz+1:] = gint_rep[nx-1:0:-1, ny-1:0:-1,nz:1:-1]

        # Tranasfer to device
        gint_rep_dev = platform.nparray_to_platform_mem(gint_rep)

        # Prepare fft plan
        fftplan = platform.plan_FFT(gint_rep_dev, axes=(0,1,2))

        # Transform the green function (in place)
        fftplan.transform(gint_rep_dev)

        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self._workspace_dev = workspace_dev
        self._gint_rep_transf_dev = gint_rep_dev
        self.fftplan = fftplan

    def solve(self, rho):
        #The transforms are done in place
        self._workspace_dev[:,:,:] = 0. # reset
        self._workspace_dev[:self.nx, :self.ny, :self.nz] = rho
        self.fftplan.transform(self._workspace_dev) # rho_rep_hat
        self._workspace_dev[:,:,:] = (self._workspace_dev
                        * self._gint_rep_transf_dev) # phi_rep_hat
        self.fftplan.itransform(self._workspace_dev) #phi_rep
        return self._workspace_dev.real[:self.nx, :self.ny, :self.nz]

def primitive_func_3d(x,y,z):
    abs_r = np.sqrt(x * x + y * y + z * z)
    inv_abs_r = 1./abs_r
    res = 1./(4*pi*epsilon_0)*(
            -0.5 * (z*z * np.arctan(x*y*inv_abs_r/z)
                    + y*y * np.arctan(x*z*inv_abs_r/y)
                    + x*x * np.arctan(y*z*inv_abs_r/x))
               + y*z*np.log(x+abs_r)
               + x*z*np.log(y+abs_r)
               + x*y*np.log(z+abs_r))
    return res
