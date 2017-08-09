/*
   GPU Kernels for the particles to mesh functions including meshing part
   @author: Adrian Oeftiger
*/

#include <cuda.h>

// implementation from: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicadd
// very slow, for <NVIDIA P100 purposes where double atomicAdd does not exist yet
// 2017-05-22 edit based on:
// http://stackoverflow.com/questions/39274472/error-function-atomicadddouble-double-has-already-been-defined

/*
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600

#else
static __inline__ __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    if (val==0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
*/

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    if (val==0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
#endif


extern "C" {

// double precision variants

__global__ void p2m_rectmesh2d_64atomics(
        // INPUTS:
          // length of x, y arrays
        const int nparticles,
          // particle positions
        double* x, double* y,
          // mesh origin
        const double x0, const double y0,
          // mesh distances per cell
        const double dx, const double dy,
          // mesh dimension (number of cells)
        const int nx, const int ny,
        // OUTPUTS:
        double *grid1d
) {
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;

    // indices
    int jx = floor((x[pidx] - x0) / dx);
    int ix = floor((y[pidx] - y0) / dy);

    // distances
    double dxi = x[pidx] - (x0 + jx * dx);
    double dyi = y[pidx] - (y0 + ix * dy);

    // weights
    double wij =    (1.-dxi/dx)*(1.-dyi/dy);
    double wi1j =   (1.-dxi/dx)*(dyi/dy)   ;
    double wij1 =   (dxi/dx)   *(1.-dyi/dy);
    double wi1j1 =  (dxi/dx)   *(dyi/dy)   ;

    if (pidx < nparticles) {
        if (jx >= 0 && jx < nx - 1 && ix >= 0 && ix < ny - 1)
        {
            atomicAdd(&grid1d[jx + ix*nx], wij);
            atomicAdd(&grid1d[jx+1 + ix*nx], wij1);
            atomicAdd(&grid1d[jx + (ix+1)*nx], wi1j);
            atomicAdd(&grid1d[jx+1 + (ix+1)*nx], wi1j1);
        }
    }
}

__global__ void p2m_rectmesh3d_64atomics(
        // INPUTS:
          // length of x, y, z arrays
        const int nparticles,
          // particle positions
        double* x, double* y, double* z,
          // mesh origin
        const double x0, const double y0, const double z0,
          // mesh distances per cell
        const double dx, const double dy, const double dz,
          // mesh dimension (number of cells)
        const int nx, const int ny, const int nz,
        // OUTPUTS:
        double *grid1d
) {
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    // indices
    int jx = floor((x[pidx] - x0) / dx);
    int ix = floor((y[pidx] - y0) / dy);
    int kx = floor((z[pidx] - z0) / dz);

    // distances
    double dxi = x[pidx] - (x0 + jx * dx);
    double dyi = y[pidx] - (y0 + ix * dy);
    double dzi = z[pidx] - (z0 + kx * dz);

    // weights
    double wijk =    (1.-dxi/dx)*(1.-dyi/dy)*(1.-dzi/dz);
    double wi1jk =   (1.-dxi/dx)*(dyi/dy)   *(1.-dzi/dz);
    double wij1k =   (dxi/dx)   *(1.-dyi/dy)*(1.-dzi/dz);
    double wi1j1k =  (dxi/dx)   *(dyi/dy)   *(1.-dzi/dz);
    double wijk1 =   (1.-dxi/dx)*(1.-dyi/dy)*(dzi/dz);
    double wi1jk1 =  (1.-dxi/dx)*(dyi/dy)   *(dzi/dz);
    double wij1k1 =  (dxi/dx)   *(1.-dyi/dy)*(dzi/dz);
    double wi1j1k1 = (dxi/dx)   *(dyi/dy)   *(dzi/dz);

    if (pidx < nparticles) {
        if (jx >= 0 && jx < nx - 1 && ix >= 0 && ix < ny - 1 && kx >= 0 && kx < nz - 1)
        {
            atomicAdd(&grid1d[jx   + ix*nx     + kx*nx*ny],     wijk);
            atomicAdd(&grid1d[jx+1 + ix*nx     + kx*nx*ny],     wij1k);
            atomicAdd(&grid1d[jx   + (ix+1)*nx + kx*nx*ny],     wi1jk);
            atomicAdd(&grid1d[jx+1 + (ix+1)*nx + kx*nx*ny],     wi1j1k);
            atomicAdd(&grid1d[jx   + ix*nx     + (kx+1)*nx*ny], wijk1);
            atomicAdd(&grid1d[jx+1 + ix*nx     + (kx+1)*nx*ny], wij1k1);
            atomicAdd(&grid1d[jx   + (ix+1)*nx + (kx+1)*nx*ny], wi1jk1);
            atomicAdd(&grid1d[jx+1 + (ix+1)*nx + (kx+1)*nx*ny], wi1j1k1);
        }
    }
}


// single precision variants

__global__ void p2m_rectmesh2d_32atomics(
        // INPUTS:
          // length of x, y arrays
        const int nparticles,
          // particle positions
        double* x, double* y,
          // mesh origin
        const double x0, const double y0,
          // mesh distances per cell
        const double dx, const double dy,
          // mesh dimension (number of cells)
        const int nx, const int ny,
        // OUTPUTS:
        float *grid1d
) {
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;

    // indices
    int jx = floor((x[pidx] - x0) / dx);
    int ix = floor((y[pidx] - y0) / dy);

    // distances
    double dxi = x[pidx] - (x0 + jx * dx);
    double dyi = y[pidx] - (y0 + ix * dy);

    // weights
    float wij =    (1.-dxi/dx)*(1.-dyi/dy);
    float wi1j =   (1.-dxi/dx)*(dyi/dy)   ;
    float wij1 =   (dxi/dx)   *(1.-dyi/dy);
    float wi1j1 =  (dxi/dx)   *(dyi/dy)   ;

    if (pidx < nparticles) {
        if (jx >= 0 && jx < nx - 1 && ix >= 0 && ix < ny - 1)
        {
            atomicAdd(&grid1d[jx + ix*nx], wij);
            atomicAdd(&grid1d[jx+1 + ix*nx], wij1);
            atomicAdd(&grid1d[jx + (ix+1)*nx], wi1j);
            atomicAdd(&grid1d[jx+1 + (ix+1)*nx], wi1j1);
        }
    }
}

__global__ void p2m_rectmesh3d_32atomics(
        // INPUTS:
          // length of x, y, z arrays
        const int nparticles,
          // particle positions
        double* x, double* y, double* z,
          // mesh origin
        const double x0, const double y0, const double z0,
          // mesh distances per cell
        const double dx, const double dy, const double dz,
          // mesh dimension (number of cells)
        const int nx, const int ny, const int nz,
        // OUTPUTS:
        float *grid1d
) {
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    // indices
    int jx = floor((x[pidx] - x0) / dx);
    int ix = floor((y[pidx] - y0) / dy);
    int kx = floor((z[pidx] - z0) / dz);

    // distances
    double dxi = x[pidx] - (x0 + jx * dx);
    double dyi = y[pidx] - (y0 + ix * dy);
    double dzi = z[pidx] - (z0 + kx * dz);

    // weights
    float wijk =    (1.-dxi/dx)*(1.-dyi/dy)*(1.-dzi/dz);
    float wi1jk =   (1.-dxi/dx)*(dyi/dy)   *(1.-dzi/dz);
    float wij1k =   (dxi/dx)   *(1.-dyi/dy)*(1.-dzi/dz);
    float wi1j1k =  (dxi/dx)   *(dyi/dy)   *(1.-dzi/dz);
    float wijk1 =   (1.-dxi/dx)*(1.-dyi/dy)*(dzi/dz);
    float wi1jk1 =  (1.-dxi/dx)*(dyi/dy)   *(dzi/dz);
    float wij1k1 =  (dxi/dx)   *(1.-dyi/dy)*(dzi/dz);
    float wi1j1k1 = (dxi/dx)   *(dyi/dy)   *(dzi/dz);

    if (pidx < nparticles) {
        if (jx >= 0 && jx < nx - 1 && ix >= 0 && ix < ny - 1 && kx >= 0 && kx < nz - 1)
        {
            atomicAdd(&grid1d[jx   + ix*nx     + kx*nx*ny],     wijk);
            atomicAdd(&grid1d[jx+1 + ix*nx     + kx*nx*ny],     wij1k);
            atomicAdd(&grid1d[jx   + (ix+1)*nx + kx*nx*ny],     wi1jk);
            atomicAdd(&grid1d[jx+1 + (ix+1)*nx + kx*nx*ny],     wi1j1k);
            atomicAdd(&grid1d[jx   + ix*nx     + (kx+1)*nx*ny], wijk1);
            atomicAdd(&grid1d[jx+1 + ix*nx     + (kx+1)*nx*ny], wij1k1);
            atomicAdd(&grid1d[jx   + (ix+1)*nx + (kx+1)*nx*ny], wi1jk1);
            atomicAdd(&grid1d[jx+1 + (ix+1)*nx + (kx+1)*nx*ny], wi1j1k1);
        }
    }
}


} /* end extern C */
