'''
FFT Poisson solvers for PyPIC
@author Stefan Hegglin, Adrian Oeftiger, Giovanni Iadarola
Implementation/Logic: Giovanni Idadarola
New interface: Stefan Hegglin, Adrian Oeftiger
'''

from __future__ import division

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl
from scipy.constants import epsilon_0

from poisson_solver import PoissonSolver
from FD_solver import compute_new_mesh_properties

try:
    from pycuda import gpuarray
    import pycuda.driver as drv
    import scikits.cuda.fft as cu_fft

except ImportError:
    print('GPU libraries (pycuda, scikits.cuda.fft) not found. GPU functionality ' +
          'not available.')


def get_Memcpy3D_d2d(src, dst, src_pitch, dst_pitch, dim_args):
    ''' Wrapper for the pycuda.driver.Memcpy3d() function (same args)
    Returns a callable object which copies the arrays on invocation of ()
    dim_args: list, [width, height, depth] !not width_in_bytes
    '''
    width, height, depth = dim_args
    width_in_bytes = width * src.strides[2]
    cpy = drv.Memcpy3D()
    cpy.set_src_device(src.ptr)
    cpy.set_dst_device(dst.ptr)
    cpy.height = np.int64(height)
    cpy.width_in_bytes = np.int64(width_in_bytes)
    cpy.depth = np.int64(depth)
    cpy.src_pitch = src_pitch
    cpy.dst_pitch = dst_pitch
    cpy.src_height = np.int64(src.shape[1])
    cpy.dst_height = np.int64(dst.shape[1])
    return cpy

def get_Memcpy2D_d2d(src, dst, src_pitch, dst_pitch, dim_args):
    ''' Wrapper for the pycuda.driver.Memcpy2d() function (same args)
    Returns a callable object which copies the arrays on invocation of ()
    dim_args: list, [width, height, depth] !not width_in_bytes
    '''
    width, height = dim_args
    width_in_bytes = width * src.strides[1]
    cpy = drv.Memcpy2D()
    cpy.set_src_device(src.gpudata)
    cpy.set_dst_device(dst.gpudata)
    cpy.height = np.int64(height)
    cpy.width_in_bytes = np.int64(width_in_bytes)
    cpy.src_pitch = src_pitch
    cpy.dst_pitch = dst_pitch
    def _copy():
        ''' Wrap the call to pass aligned=True which seems to be necessary
        in the 2D version (compared to 3D where it doesn't work with this arg
        '''
        cpy(aligned=True)
    return _copy

class GPUFFTPoissonSolver(PoissonSolver):
    '''
    FFT openboundary solver on the GPU using the integrated Green's function
    The class works for 2 and 3 dimensional systems.
    The corresponding greens functions/ algorithms are set (monkey patching)
    during the initialization of the class and depend on the dimension of the
    mesh.
    '''
    def __init__(self, mesh):
        '''
        Args:
            mesh The mesh on which the solver will operate. The dimensionality
                 is deducted from mesh.dimension
        '''
        # create the mesh grid and compute the greens function on it
        self.mesh = mesh
        mesh_shape = self.mesh.shape # nx, ny, (nz)
        mesh_shape2 = [2*n for n in mesh_shape] # 2*nx, 2*ny, (2*nz)
        mesh_distances = self.mesh.distances
        #preallocate for fast d2d copies
        self._rho = gpuarray.empty(list(reversed(mesh_shape)),
                    dtype=np.complex128)
        self.fgreentr = gpuarray.empty(list(reversed(mesh_shape2)),
                        dtype=np.complex128)
        self.tmpspace = gpuarray.zeros_like(self.fgreentr)
        sizeof_complex = np.dtype(np.complex128).itemsize
 
        # dimensionality function dispatch
        dim = self.mesh.dimension
        self._fgreen = getattr(self, '_fgreen' + str(dim) + 'd')
        self._mirror = getattr(self, '_mirror' + str(dim) + 'd')
        copy_fn = {'3d' : get_Memcpy3D_d2d, '2d': get_Memcpy2D_d2d}
        memcpy_nd = copy_fn[str(dim) + 'd']
        dim_args = self.mesh.shape
        self._cpyrho2tmp = memcpy_nd(
            src=self._rho, dst=self.tmpspace,
            src_pitch=self.mesh.nx*sizeof_complex,
            dst_pitch=2*self.mesh.nx*sizeof_complex,
            dim_args=dim_args)
        self._cpytmp2rho = memcpy_nd(
            src=self.tmpspace, dst=self._rho,
            src_pitch=2*self.mesh.nx*sizeof_complex,
            dst_pitch=self.mesh.nx*sizeof_complex,
            dim_args=dim_args)

        mesh_arr = [-mesh_distances[i]/2 + np.arange(mesh_shape[i]+1)
                                            * mesh_distances[i]
                    for i in xrange(self.mesh.dimension)
                   ]
        mesh_grids = np.meshgrid(*list(reversed(mesh_arr)), indexing='ij')
        fgreen = self._fgreen(*mesh_grids)
        fgreen = self._mirror(fgreen)
        self.plan_forward = cu_fft.Plan(self.tmpspace.shape, in_dtype=np.complex128,
                                        out_dtype=np.complex128)
        self.plan_backward = cu_fft.Plan(self.tmpspace.shape, in_dtype=np.complex128,
                                         out_dtype=np.complex128)
        cu_fft.fft(gpuarray.to_gpu(fgreen), self.fgreentr, plan=self.plan_forward)

    def poisson_solve(self, rho):
        ''' Solve the poisson equation with the given charge distribution
        Args:
            rho: Charge distribution (same dimensions as mesh)
        Returns:
            Phi (same dimensions as rho)
        '''
        drv.memcpy_dtod(self._rho.gpudata, rho.astype(np.complex128).gpudata,
                        self._rho.nbytes)
        # set to 0 since it might be filled with the old potential
        self.tmpspace.fill(0)
        mat = self._rho
        target = self.tmpspace
        self._cpyrho2tmp()
        cu_fft.fft(self.tmpspace, self.tmpspace, plan=self.plan_forward)
        cu_fft.ifft(self.tmpspace * self.fgreentr, self.tmpspace,
                    plan=self.plan_backward)
        # store the result in the rho gpuarray to save space
        self._cpytmp2rho()
        self._buf = self._rho.get()
        phi = self._rho.real/(2**self.mesh.dimension *self.mesh.n_nodes) # scale (cuFFT is unscaled)
        phi *= self.mesh.volume_elem/(2**(self.mesh.dimension-1)*np.pi*epsilon_0)
        return phi



    def _mirror2d(self, green):
        ''' Mirror the greens function in the big domain
        The area in the lower left (:ny, :nx) must contain the greens function
        green:  (2*ny, 2*nx) shaped array
        returns: the same array (in place transformation)
        '''
        nx = self.mesh.nx
        ny = self.mesh.ny
        green[ny:, :nx]  = green[ ny:0:-1, :nx]
        green[:ny,  nx:] = green[:ny,       nx:0:-1]
        green[ny:,  nx:] = green[ ny:0:-1,  nx:0:-1]
        return green

    def _mirror3d(self, green):
        ''' Mirror the greens function in the big domain
        The area in the lower left front (:nz, :ny, :nx) must contain the
        greens function
        green:  (2*nz, 2*ny, 2*nx) shaped array
        returns: the same array (in place transformation)
        '''
        nz = self.mesh.nz
        ny = self.mesh.ny
        nx = self.mesh.nx
        green[nz:, :ny, :nx] = green[nz:0:-1,  :ny,      :nx]
        green[:nz, ny:, :nx] = green[:nz,       ny:0:-1, :nx]
        green[nz:, ny:, :nx] = green[nz:0:-1,   ny:0:-1, :nx]
        green[:nz, :ny, nx:] = green[:nz,      :ny,       nx:0:-1]
        green[nz:, :ny, nx:] = green[nz:0:-1,  :ny,       nx:0:-1]
        green[:nz, ny:, nx:] = green[:nz,       ny:0:-1,  nx:0:-1]
        green[nz:, ny:, nx:] = green[nz:0:-1,   ny:0:-1,  nx:0:-1]
        return green

    def _fgreen2d(self, x, y):
        '''
        Return the periodic integrated greens funcion on the 'original'
        domain
        J. Qiang, M. A. Furman, and R.D. Ryne, J. Comput. Phys 198, 278 (2004)
        [times a factor of -1/2 !?]
        Args:
            x,y: arrays, e.g. x,y = np.meshgrid(xx,yy)
        '''
        abs_r = np.sqrt(x * x + y * y)
        inv_abs_r = 1./abs_r
        tmpfgreen = -1./2*(-3*x*y + x*x*np.arctan(y/x)
                     + y*y*np.arctan(x/y) + x*y*np.log(x*x + y*y)
                    )
        fgreen = np.zeros((2 * self.mesh.ny,
                           2 * self.mesh.nx), dtype=np.complex128)
        # evaluate the indefinite integral per cell (int_a^b f = F(b) - F(a))
        fgreen[:self.mesh.ny, :self.mesh.nx] = (
                  tmpfgreen[1:, 1:]   - tmpfgreen[1:, :-1]
                + tmpfgreen[:-1, :-1] - tmpfgreen[:-1, 1:]
                ) * 1./self.mesh.volume_elem # divide by vol_elem to average!
        return fgreen



    def _fgreen3d(self, x, y, z):
        ''' Return the periodic integrated greens funcion on the 'original'
        domain
        Qiang, Lidia, Ryne,Limborg-Deprey, PRSTAB 10, 129901 (2007)
        Args:
            x,y,z: arrays, e.g. x, y, z = np.meshgrid(xx, yy, zz)
        '''
        abs_r = np.sqrt(x * x + y * y + z * z)
        inv_abs_r = 1./abs_r
        tmpfgreen =  (-(  +    z*z * np.arctan(x*y*inv_abs_r/z)
                      +   y*y * np.arctan(x*z*inv_abs_r/y)
                      +   x*x * np.arctan(y*z*inv_abs_r/x)
                   )/2.
                    + y*z*np.log(x+abs_r)
                    + x*z*np.log(y+abs_r)
                    + x*y*np.log(z+abs_r))
        fgreen = np.zeros((2 * self.mesh.nz,
                           2 * self.mesh.ny,
                           2 * self.mesh.nx), dtype=np.complex128)
        # evaluate the indefinite integral per cell (int_a^b f = F(b) - F(a))
        fgreen[:self.mesh.nz, :self.mesh.ny, :self.mesh.nx] = (
                 tmpfgreen[ 1:,  1:,  1:]
                -tmpfgreen[:-1,  1:,  1:]
                -tmpfgreen[ 1:, :-1,  1:]
                +tmpfgreen[:-1, :-1,  1:]
                -tmpfgreen[ 1:,  1:, :-1]
                +tmpfgreen[:-1,  1:, :-1]
                +tmpfgreen[ 1:, :-1, :-1]
                -tmpfgreen[:-1, :-1, :-1]
                ) * 1./self.mesh.volume_elem # divide by vol_elem to average!
        return fgreen


#class GPUFFTPoissonSolverNonIGF3D(GPUFFTPoissonSolver):
#    """
#    FFT openboundary solver on the GPU NOT using the integrated Greens function
#    Do not use this solver for high aspect ratios of the mesh/beam!
#    The only difference to the super class is the _green() function, which
#    automatically gets called when the __init__ of the base class is called.
#    """
#    def __init__(self, mesh):
#        '''
#        mesh:          3d  mesh on which the operator operates
#        '''
#        super(GPUFFTPoissonSolverNonIGF3D, self).__init__(mesh)
#
#
#    def _green(self, x, y, z):
#        ''' Return the greens function (-1/r) evaluated on x, y, z, returns
#        a 8x bigger domain with the greens function in the 'original' part
#        '''
#        abs_r = np.sqrt(x * x + y * y + z * z)
#        green = 1./abs_r
#        fgreen = np.zeros((2 * self.mesh.nz,
#                           2 * self.mesh.ny,
#                           2 * self.mesh.nx), dtype=np.complex128)
#        fgreen[:self.mesh.nz, :self.mesh.ny, :self.mesh.nx] = green[1:, 1:, 1:]
#        return fgreen

################################ Old implementations / Wrappers for them ######

class FFT_OpenBoundary_SquareGrid(PoissonSolver):
    '''
    Wrapper for the old PyPIC FFT open boundary solver
    '''
    def __init__(self, x_aper, y_aper, Dh, fftlib='pyfftw'):
        na = lambda x:np.array([x])
        params = compute_new_mesh_properties(
                     x_aper, y_aper, Dh, ext_boundary=False) #change to true for bw-compatibility

        self.Dh, self.xg, self.Nxg, self.bias_x, self.yg, self.Nyg, self.bias_y = params
        dx = self.xg[1] - self.xg[0]
        dy = self.yg[1] - self.yg[0]

        nx = len(self.xg)
        ny = len(self.yg)
        mx = -dx / 2 + np.arange(nx + 1) * dx
        my = -dy / 2 + np.arange(ny + 1) * dy
        x, y = np.meshgrid(mx, my)
        r2 = x ** 2 + y ** 2
        # Antiderivative
        tmpfgreen = -1 / 2 * (-3 * x * y + x * y * np.log(r2)
                  + x * x * np.arctan(y / x) + y * y * np.arctan(x / y)) # * 2 / dx / dy

        fgreen = np.zeros((2 * ny, 2 * nx))
        # Integration and circular Green's function
        fgreen[:ny, :nx] = tmpfgreen[1:, 1:] + tmpfgreen[:-1, :-1] - tmpfgreen[1:, :-1] - tmpfgreen[:-1, 1:]
        fgreen[ny:, :nx] = fgreen[ny:0:-1, :nx]
        fgreen[:ny, nx:] = fgreen[:ny, nx:0:-1]
        fgreen[ny:, nx:] = fgreen[ny:0:-1, nx:0:-1]

        if fftlib == 'pyfftw':
            try:
                import pyfftw
                print 'Using PyFFTW'
                #prepare fftw's
                tmprho = fgreen.copy()
                fft_first = pyfftw.builders.fft(tmprho[:ny, :].copy(), axis = 1)
                transf1 = (fgreen*(1.+1j))*0.
                transf1[:ny, :] = fft_first(tmprho[:ny, :].copy())
                fft_second = pyfftw.builders.fft(transf1.copy(), axis = 0)
                fftphi_new = fft_second(transf1.copy())* fgreen
                ifft_first = pyfftw.builders.ifft(fftphi_new.copy(), axis = 0)
                itransf1 = ifft_first(fftphi_new.copy())
                ifft_second = pyfftw.builders.ifft(itransf1[:ny, :].copy(), axis = 1)

                def fft2(x):
                    tmp = (x*(1.+1j))*0.
                    tmp[:ny, :] = fft_first(x[:ny, :])
                    return fft_second(tmp)

                def ifft2(x):
                    tmp = ifft_first(x)
                    res = 0*x
                    res[:ny, :] = np.real(ifft_second(tmp[:ny, :]))
                    return res

                self.fft2 = fft2
                self.ifft2 = ifft2

            except ImportError as err:
                print 'Failed to import pyfftw'
                print 'Got exception: ', err
                print 'Using numpy fft'
                self.fft2 = np.fft.fft2
                self.ifft2 = np.fft.ifft2
        elif fftlib == 'numpy':
                print 'Using numpy FFT'
                self.fft2 = np.fft.fft2
                self.ifft2 = np.fft.ifft2
        else:
            raise ValueError('fftlib not recognized!')
        self.fgreen = fgreen
        self.fgreentr = np.fft.fft2(fgreen).copy()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy

    def poisson_solve(self, mesh_charges):
        tmprho = 0.*self.fgreen
        mesh_charges = mesh_charges.reshape(self.Nyg, self.Nxg) / (self.dx*self.dy)
        tmprho[:self.ny, :self.nx] = mesh_charges

        fftphi = self.fft2(tmprho) * self.fgreentr

        tmpphi = self.ifft2(fftphi)
        phi = 1./(4. * np.pi * epsilon_0)*np.real(tmpphi[:self.ny, :self.nx]).T
        phi = phi.reshape(self.Nxg, self.Nyg).T.flatten()
        return phi*2 #magic number... TODO find out why this is needed!!


class FFT_PEC_Boundary_SquareGrid(PoissonSolver):
    '''
    Wrapper for the old PyPIC FFT perdiodic boundary solver
    '''
    def __init__(self, x_aper, y_aper, Dh, fftlib='pyfftw'):
        na = lambda x:np.array([x])
        params = compute_new_mesh_properties(
                     x_aper, y_aper, Dh, ext_boundary=True) #always True!

        self.Dh, self.xg, self.Nxg, self.bias_x, self.yg, self.Nyg, self.bias_y = params
        self.i_min = np.min(np.where(self.xg>=-x_aper)[0])
        self.i_max = np.max(np.where(self.xg<=x_aper)[0])+1
        self.j_min = np.min(np.where(self.yg>=-y_aper)[0])
        self.j_max = np.max(np.where(self.yg<=y_aper)[0])+1

        dummy = np.zeros((self.Nxg,self.Nyg))
        m, n = dummy[self.i_min:self.i_max,self.j_min:self.j_max].shape;
        xx = np.arange(1,m+0.5,1);
        yy = np.arange(1,n+0.5,1);

        YY, XX = np.meshgrid(yy,xx)
        self.green = 4.*epsilon_0*(np.sin(XX/2*np.pi/float(m+1.))**2/self.Dh**2+\
                     np.sin(YY/2.*np.pi/float(n+1.))**2/self.Dh**2);

        # handle border
        [xn, yn]=np.meshgrid(self.xg,self.yg)

        xn=xn.T
        xn=xn.flatten()

        yn=yn.T
        yn=yn.flatten()
        #% xn and yn are stored such that the external index is on x 

        flag_outside_n=np.logical_or(np.abs(xn)>x_aper,np.abs(yn)>y_aper)
        flag_inside_n=~(flag_outside_n)


        flag_outside_n_mat=np.reshape(flag_outside_n,(self.Nyg,self.Nxg),'F');
        flag_outside_n_mat=flag_outside_n_mat.T
        [gx,gy]=np.gradient(np.double(flag_outside_n_mat));
        gradmod=abs(gx)+abs(gy);
        flag_border_mat=np.logical_and((gradmod>0), flag_outside_n_mat);
        self.flag_border_mat = flag_border_mat

        if fftlib == 'pyfftw':
            try:
                import pyfftw
                rhocut = dummy[self.i_min:self.i_max,self.j_min:self.j_max]
                m, n = rhocut.shape;
                tmp = np.zeros((2*m + 2, n))
                self.ffti = pyfftw.builders.fft(tmp.copy(), axis=0)
                tmp = np.zeros((m, 2*n + 2))
                self.fftj = pyfftw.builders.fft(tmp.copy(), axis=1)
            except ImportError as err:
                print 'Failed to import pyfftw'
                print 'Got exception: ', err
                print 'Using numpy fft'
                self.ffti = lambda xx: np.fft.fft(xx, axis=0)
                self.fftj = lambda xx: np.fft.fft(xx, axis=1)
        elif fftlib == 'numpy':
            self.ffti = lambda xx: np.fft.fft(xx, axis=0)
            self.fftj = lambda xx: np.fft.fft(xx, axis=1)
        else:
            raise ValueError('fftlib not recognized!!!!')

    def dst2(self, x):
        m, n = x.shape;
        #transform along i
        tmp = np.zeros((2*m + 2, n))
        tmp[1:m+1, :] = x
        tmp=-(self.ffti(tmp).imag)
        xtr_i = np.sqrt(2./(m+1.))*tmp[1:m+1, :]

        #transform along j
        tmp = np.zeros((m, 2*n + 2))
        tmp[:, 1:n+1] = xtr_i
        tmp=-(self.fftj(tmp).imag)
        x_bar = np.sqrt(2./(n+1.))*tmp[:, 1:n+1]

        return x_bar

    def poisson_solve(self, mesh_charges):
        mesh_charges = mesh_charges.reshape(self.Nyg, self.Nxg).T / (self.Dh*self.Dh)
        rhocut = mesh_charges[self.i_min:self.i_max,self.j_min:self.j_max]
        rho_bar =  self.dst2(rhocut)
        phi_bar = rho_bar/self.green
        phi = np.zeros((self.Nxg, self.Nyg))
        phi[self.i_min:self.i_max,self.j_min:self.j_max] = self.dst2(phi_bar).copy()
        phi = phi.reshape(self.Nxg, self.Nyg).T.flatten()
        return phi
