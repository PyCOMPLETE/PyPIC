/* 
   GPU Kernels for the particles to mesh functions
   @author: Stefan Hegglin, Adrian Oeftiger
*/

// implementation from: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicadd
// very slow, for testing purposes
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}


extern "C" {

__global__ void particles_to_mesh_2d(double *grid1d, int stride, double *wij, double *wi1j,
                                     double *wij1, double *wi1j1, int *i, int *j)
{
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;
    int ix = i[pidx];
    int jx = j[pidx];

    atomicAdd(&grid1d[jx + ix*stride], wij[pidx]);
    atomicAdd(&grid1d[jx+1 + ix*stride], wij1[pidx]);
    atomicAdd(&grid1d[jx + (ix+1)*stride], wi1j[pidx]);
    atomicAdd(&grid1d[jx+1 + (ix+1)*stride], wi1j1[pidx]);

}

__global__ void particles_to_mesh_3d(double *grid1d, int stridex, int stridey,
                                     double *wijk, double *wi1jk, double *wij1k, double *wi1j1k,
                                     double *wijk1, double *wi1jk1, double* wij1k1, double* wi1j1k1,
                                     int *i, int *j, int* k)
{
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    int jj = blockIdx.y*blockDim.y + threadIdx.y;
    int kk = blockIdx.z;
    //int pidx = ii + jj*stridex + kk*stridex*stridey;
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;
    int ix = i[pidx];
    int jx = j[pidx];
    int kx = k[pidx];

    atomicAdd(&grid1d[jx   + ix*stridex     + kx*stridex*stridey],     wijk[pidx]);
    atomicAdd(&grid1d[jx+1 + ix*stridex     + kx*stridex*stridey],     wij1k[pidx]);
    atomicAdd(&grid1d[jx   + (ix+1)*stridex + kx*stridex*stridey],     wi1jk[pidx]);
    atomicAdd(&grid1d[jx+1 + (ix+1)*stridex + kx*stridex*stridey],     wi1j1k[pidx]);
    atomicAdd(&grid1d[jx   + ix*stridex     + (kx+1)*stridex*stridey], wijk1[pidx]);
    atomicAdd(&grid1d[jx+1 + ix*stridex     + (kx+1)*stridex*stridey], wij1k1[pidx]);
    atomicAdd(&grid1d[jx   + (ix+1)*stridex + (kx+1)*stridex*stridey], wi1jk1[pidx]);
    atomicAdd(&grid1d[jx+1 + (ix+1)*stridex + (kx+1)*stridex*stridey], wi1j1k1[pidx]);
}
} /* end extern C */ 
