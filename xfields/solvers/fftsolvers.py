from .base import Solver

class FFTSolver2D(Solver):

    def solve(self, rho):
        pass

class FFTSolver3D(Solver):

    def __init__(self, dx, dy, dz, nx, ny, nz):

        # Prepare arrays
        gint_rep = np.zeros((2*nx, 2*ny, 2*nz), dtype=np.float64, order='F')
        rho_rep = np.zeros((2*nx, 2*ny, 2*nz), dtype=np.float64, order='F')

        # Build grid for primitive function
        xg_F = np.arange(0, nx+2) * dx - dx/2
        yg_F = np.arange(0, ny+2) * dy - dy/2
        zg_F = np.arange(0, nz+2) * dz - dz/2
        XX_F, YY_F, ZZ_F = np.meshgrid(xg_F, yg_F, zg_F, indexing='ij')

        # Compute primitive
        F_temp = primitive_func_3d(XX_F, YY_F, ZZ_F)

        # Integrated Green Function
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

        # Transform the green function
        gint_rep_transf = np.fft.fftn(gint_rep)

        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self._rho_rep = rho_rep
        self._phi_rep = phi_rep

    def solve(self, rho):
        rho_rep[:,:,:] = 0. # reset
        rho_rep[:nx, :ny, :nz] = rho
        phi_rep = np.fft.ifftn(np.fft.fftn(rho_rep) * gint_rep_transf)
        return np.real(phi_rep)[:nx, :ny, :nz]

