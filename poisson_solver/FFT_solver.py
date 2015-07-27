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


def get_Memcpy3D_d2d(width_in_bytes, height, depth, src, dst,
                     src_pitch, dst_pitch):
    ''' src on host, dst on device, both 3-dimensional '''
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


class DEBUG_FFT3_OpenBoundary(PoissonSolver):
    def __init__(self, mesh, IGF=True):
        mx = -mesh.dx/2 + np.arange(mesh.nx+1) * mesh.dx
        my = -mesh.dy/2 + np.arange(mesh.ny+1) * mesh.dy
        mz = -mesh.dz/2 + np.arange(mesh.nz+1) * mesh.dz
        z, y, x = np.meshgrid(mz, my, mx, indexing='ij') #TODO check indices=..

        nx = mesh.nx
        ny = mesh.ny
        nz = mesh.nz
        self.mesh = mesh
        abs_r = np.sqrt(x * x + y * y + z * z)
        inv_abs_r = 1./abs_r
        if IGF:
            tmpfgreen = +(-(  z*z * np.arctan(x*y*inv_abs_r/z)
                          +   y*y * np.arctan(x*z*inv_abs_r/y)
                          +   x*x * np.arctan(y*z*inv_abs_r/x)
                          )/2.
                        + y*z*np.log(x+abs_r)
                        + x*z*np.log(y+abs_r)
                        + x*y*np.log(z+abs_r))
            tmpfgreen *= 1
        else:
            tmpfgreen = inv_abs_r
        fgreen = np.zeros((2 * nz, 2 * ny, 2 * nx), dtype=np.complex128) + 1000
        fgreen[:nz, :ny, :nx] =  tmpfgreen[1:, 1:, 1:]
        #fgreen[:nz, :ny, :nx] =(-tmpfgreen[ 1:,  1:,  1:]
        #                        +tmpfgreen[-1:,  1:,  1:]
        #                        +tmpfgreen[ 1:, -1:,  1:]
        #                        -tmpfgreen[-1:, -1:,  1:]
        #                        +tmpfgreen[ 1:,  1:, -1:]
        #                        -tmpfgreen[-1:,  1:, -1:]
        #                        -tmpfgreen[ 1:, -1:, -1:]
        #                        +tmpfgreen[-1:, -1:, -1:])

        #import scipy.integrate as sci
        #tmp = sci.cumtrapz(tmpfgreen, dx=1.)
        #fgreen[:nz, :ny, :nx] = tmp


        fgreen[:nz, :ny, :nx] =(
                 tmpfgreen[ 1:,  1:,  1:]
                -tmpfgreen[:-1,  1:,  1:]
                -tmpfgreen[ 1:, :-1,  1:]
                +tmpfgreen[:-1, :-1,  1:]
                -tmpfgreen[ 1:,  1:, :-1]
                +tmpfgreen[:-1,  1:, :-1]
                +tmpfgreen[ 1:, :-1, :-1]
                -tmpfgreen[:-1, :-1, :-1])
        # mirror the artificially added regions
        fgreen[nz:, :ny, :nx] = fgreen[nz:0:-1,  :ny,      :nx]
        fgreen[:nz, ny:, :nx] = fgreen[:nz,       ny:0:-1, :nx]
        fgreen[nz:, ny:, :nx] = fgreen[nz:0:-1,   ny:0:-1, :nx]
        fgreen[:nz, :ny, nx:] = fgreen[:nz,      :ny,       nx:0:-1]
        fgreen[nz:, :ny, nx:] = fgreen[nz:0:-1,  :ny,       nx:0:-1]
        fgreen[:nz, ny:, nx:] = fgreen[:nz,       ny:0:-1,  nx:0:-1]
        fgreen[nz:, ny:, nx:] = fgreen[nz:0:-1,   ny:0:-1,  nx:0:-1]
        self.fgreentr = np.fft.fftn(fgreen,s=fgreen.shape)
        self.nx = nx
        self.ny = ny
        self.nz = nz


    def poisson_solve(self, rho):
        ''' Solve the poisson equation using hockney's algorithm:
            phi = ifft(fft(rho*green))
            fft/ifft are in place 2d-C2C-fft using cuFFT
        '''
        rho = rho.get().astype(np.complex128)
        tmp = np.zeros((2*self.nz, 2*self.ny, 2*self.nx), dtype=np.complex128)
        tmp[:self.nz, :self.ny, :self.nx] = rho
        phi = np.fft.ifftn(np.fft.fftn(tmp, tmp.shape)*self.fgreentr)
        phi = np.real(phi[:self.nz, :self.ny, :self.nx]).copy()
        phi *= 1./(4*np.pi*epsilon_0)
        phi_gpu = gpuarray.zeros(phi.shape, dtype=np.float64)
        phi_gpu.set(phi)
        return phi_gpu



class GPU_FFT_OpenBoundary(PoissonSolver):
    """
    FFT openboundary solver on the GPU

    3d integrated greens function:
    Qiang, Lidia, Ryne,Limborg-Deprey, PRSTAB 10, 129901 (2007)
    Erratum: Three-dimensional quasistatic model
    for high brightness beam dynamics simulation[PRSTAB 9, 044204 (2006)]
    """
    def __init__(self, mesh, IGF=False):
        '''
        mesh:           mesh on which the operator operates
        free_memory:    flag determining whether the memory on the GPU should
                        be freed if possible after each call to solve
        IGF:            Use integrated greens function (True/False)
        '''
        mx = -mesh.dx/2 + np.arange(mesh.nx+1) * mesh.dx
        my = -mesh.dy/2 + np.arange(mesh.ny+1) * mesh.dy
        mz = -mesh.dz/2 + np.arange(mesh.nz+1) * mesh.dz
        z, y, x = np.meshgrid(mz, my, mx, indexing='ij') #TODO check indices=..
        nx = mesh.nx
        ny = mesh.ny
        nz = mesh.nz
        self.mesh = mesh
        ### define the 3d free space green function
        #abs_r = np.sqrt(mesh.dx*mesh.dx*x * x + mesh.dy*mesh.dy*y * y + mesh.dz*mesh.dz*z * z)
        abs_r = np.sqrt(x * x + y * y + z * z)
        inv_abs_r = 1./abs_r#**np.sqrt(2)
        if IGF:
            tmpfgreen = +(-(  z*z * np.arctan(x*y*inv_abs_r/z)
                          +   y*y * np.arctan(x*z*inv_abs_r/y)
                          +   x*x * np.arctan(y*z*inv_abs_r/x)
                          )/2.
                        + y*z*np.log(x+abs_r)
                        + x*z*np.log(y+abs_r)
                        + x*y*np.log(z+abs_r))
        else:
            tmpfgreen = inv_abs_r

        fgreen = np.zeros((2 * nz, 2 * ny, 2 * nx), dtype=np.complex128)
        fgreen[:nz, :ny, :nx] =(
                 tmpfgreen[ 1:,  1:,  1:]
                -tmpfgreen[:-1,  1:,  1:]
                -tmpfgreen[ 1:, :-1,  1:]
                +tmpfgreen[:-1, :-1,  1:]
                -tmpfgreen[ 1:,  1:, :-1]
                +tmpfgreen[:-1,  1:, :-1]
                +tmpfgreen[ 1:, :-1, :-1]
                -tmpfgreen[:-1, :-1, :-1])
        # mirror the artificially added regions
        fgreen[nz:, :ny, :nx] = fgreen[nz:0:-1,  :ny,      :nx]
        fgreen[:nz, ny:, :nx] = fgreen[:nz,       ny:0:-1, :nx]
        fgreen[nz:, ny:, :nx] = fgreen[nz:0:-1,   ny:0:-1, :nx]
        fgreen[:nz, :ny, nx:] = fgreen[:nz,      :ny,       nx:0:-1]
        fgreen[nz:, :ny, nx:] = fgreen[nz:0:-1,  :ny,       nx:0:-1]
        fgreen[:nz, ny:, nx:] = fgreen[:nz,       ny:0:-1,  nx:0:-1]
        fgreen[nz:, ny:, nx:] = fgreen[nz:0:-1,   ny:0:-1,  nx:0:-1]
        self.fgreentr = gpuarray.empty(fgreen.shape, dtype=np.complex128)
        self.tmpspace = gpuarray.zeros_like(self.fgreentr)

        self.plan_forward = cu_fft.Plan(self.tmpspace.shape, in_dtype=np.complex128,
                                        out_dtype=np.complex128)
        self.plan_backward = cu_fft.Plan(self.tmpspace.shape, in_dtype=np.complex128,
                                         out_dtype=np.complex128)
        cu_fft.fft(gpuarray.to_gpu(fgreen), self.fgreentr, plan=self.plan_forward)
        self.nx = nx
        self.ny = ny
        self.nz = nz


    def poisson_solve(self, rho):
        ''' Solve the poisson equation using hockney's algorithm:
            phi = ifft(fft(rho*green))
            fft/ifft are in place 2d-C2C-fft using cuFFT
        '''
        rho = rho.astype(np.complex128)
        # set to 0 since it might be filled with the old potential
        self.tmpspace.fill(0)
        sizeof_complex = np.dtype(np.complex128).itemsize
        copy_d2d_rho2tmp = get_Memcpy3D_d2d(
                width_in_bytes=self.nx*sizeof_complex,
                height=self.ny, depth=self.nz, src=rho, dst=self.tmpspace,
                src_pitch=self.nx*sizeof_complex, dst_pitch=self.tmpspace.strides[1])
        copy_d2d_tmp2rho = get_Memcpy3D_d2d(
                width_in_bytes=self.nx*sizeof_complex,
                height=self.ny, depth=self.nz,
                src=self.tmpspace, dst=rho,
                src_pitch=self.tmpspace.strides[1],
                dst_pitch=self.nx*sizeof_complex)
        copy_d2d_rho2tmp()
        cu_fft.fft(self.tmpspace, self.tmpspace, plan=self.plan_forward)
        cu_fft.ifft(self.tmpspace * self.fgreentr, self.tmpspace,
                    plan=self.plan_backward)

        # store the result in the rho gpuarray to save space
        copy_d2d_tmp2rho()
        phi = rho.real/(8.*self.mesh.n_nodes) # scale (cuFFT is unscaled) and real()
        phi *= 1./(4*np.pi*epsilon_0)
        return phi


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
