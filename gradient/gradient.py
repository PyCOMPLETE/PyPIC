import numpy as np
from scipy.constants import e

import os
where = os.path.dirname(os.path.abspath(__file__)) + '/'

try:
    from pycuda import driver as cuda
    from pycuda import gpuarray
    from pycuda.compiler import SourceModule
    from pycuda.tools import DeviceData
except ImportError:
    print('pycuda not found. no gpu capabilities will be available')


def make_GPU_gradient(mesh, context):
    '''Prepare to compute gradient on the GPU w.r.t. the given mesh.
    Return gradient function.
    '''
    mx = int(getattr(mesh, 'nx', 1))
    my = int(getattr(mesh, 'ny', 1))
    mz = int(getattr(mesh, 'nz', 1))
    # assert that mx, my are powers of 2
    assert mx != 0 and ((mx & (mx - 1)) == 0)
    assert my != 0 and ((my & (my - 1)) == 0)
    # assert mz != 0 and ((mz & (mz - 1)) == 0) not needed in z direction
    # since we always split into blocks of 1 in z direction

    dxInv = np.array(1./getattr(mesh, 'dx', 1), dtype=np.float64)
    dyInv = np.array(1./getattr(mesh, 'dy', 1), dtype=np.float64)
    dzInv = np.array(1./getattr(mesh, 'dz', 1), dtype=np.float64)

    sizeof_double = 8
    with open(where + 'gradient2.cu') as fdlib:
        source = fdlib.read()
    module = SourceModule(source)

    mx_ptr = module.get_global("mx")[0]
    my_ptr = module.get_global("my")[0]
    mz_ptr = module.get_global("mz")[0]
    cuda.memcpy_htod(mx_ptr, np.array(mx, dtype=np.int32))
    cuda.memcpy_htod(my_ptr, np.array(my, dtype=np.int32))
    cuda.memcpy_htod(mz_ptr, np.array(mz, dtype=np.int32))

    dxInv_ptr = module.get_global("dxInv")[0]
    dyInv_ptr = module.get_global("dyInv")[0]
    dzInv_ptr = module.get_global("dzInv")[0]
    cuda.memcpy_htod(dxInv_ptr, dxInv)
    cuda.memcpy_htod(dyInv_ptr, dyInv)
    cuda.memcpy_htod(dzInv_ptr, dzInv)

    deriv_x = module.get_function("gradient_x")
    deriv_y = module.get_function("gradient_y")
    deriv_z = module.get_function("gradient_z")

    block, grid = mesh.get_domain_decomposition(DeviceData().max_threads)

    d_deriv_x = gpuarray.empty(shape=(1, mesh.n_nodes), dtype=np.float64)
    d_deriv_y = gpuarray.empty_like(d_deriv_x)
    d_deriv_z = gpuarray.empty_like(d_deriv_x)

    def _gradient(scalar_values):
        '''Calculate three-dimensional gradient for GPUArray
        scalar_values.
        '''
        deriv_x(scalar_values, d_deriv_x, block=block, grid=grid)
        deriv_y(scalar_values, d_deriv_y, block=block, grid=grid)
        deriv_z(scalar_values, d_deriv_z, block=block, grid=grid)
        context.synchronize()

        return (d_deriv_x, d_deriv_y, d_deriv_z)[:mesh.dimension]
    return _gradient

def numpy_gradient(mesh):
    ''' Return a gradient function'''
    def _gradient(scalar_values):
        '''Return the gradient of the scalar_values on the mesh'''
        D = np.gradient(-scalar_values.reshape(mesh.shape),
                        *(list(reversed(mesh.distances))))
        return list(reversed(D))
    return _gradient

