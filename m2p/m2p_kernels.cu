/* 
   GPU Kernels for the mesh to particles functions
   @author: Stefan Hegglin, Adrian Oeftiger
*/

extern "C" {

__global__ void mesh_to_particles_2d(double* particles_quantity, double *mesh_quantity,
                                     const int stridex,
                                     double *wij, double *wi1j, double *wij1, double *wi1j1,
                                     int *i, int *j)
{
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;
    int ix = i[pidx];
    int jx = j[pidx];

    particles_quantity[pidx] = ( wij[pidx]   * mesh_quantity[jx   + ix*stridex    ]
                               + wij1[pidx]  * mesh_quantity[jx+1 + ix*stridex    ]
                               + wi1j[pidx]  * mesh_quantity[jx+  + (ix+1)*stridex]
                               + wi1j1[pidx] * mesh_quantity[jx+1 + (ix+1)*stridex]);
}

__global__ void field_to_particles_2d(double* forcex, double* forcey, double* fieldx, double* fieldy,
                                      const int stride, double *wij, double *wi1j, double *wij1, double *wi1j1, int *i, int *j)
{
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;
    int jx = j[pidx];
    int ix = i[pidx];
    forcex[pidx] =    ( wij[pidx] *  fieldx[jx + ix*stride]
                      + wij1[pidx] * fieldx[jx+1 + ix*stride]
                      + wi1j[pidx] * fieldx[jx + (ix+1)*stride]
                      + wi1j1[pidx] *fieldx[jx+1 + (ix+1)*stride]);
    forcey[pidx] =    ( wij[pidx] *  fieldy[jx + ix*stride]
                      + wij1[pidx] * fieldy[jx+1 + ix*stride]
                      + wi1j[pidx] * fieldy[jx + (ix+1)*stride]
                      + wi1j1[pidx] *fieldy[jx+1 + (ix+1)*stride]);
}

__global__ void field_to_particles_3d(double* forcex, double* forcey, double* forcez,
                                      double* fieldx, double* fieldy, double* fieldz,
                                      const int stridex, const int stridey,
                                      double *wijk, double *wi1jk, double *wij1k, double *wi1j1k,
                                      double *wijk1, double *wi1jk1, double* wij1k1, double* wi1j1k1,
                                      int *i, int *j, int* k)
{
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    int jj = blockIdx.y*blockDim.y + threadIdx.y;
    int kk = blockIdx.z;
   // int pidx = ii + jj*stridex + kk*stridex*stridey;
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;
    int ix = i[pidx];
    int jx = j[pidx];
    int kx = k[pidx];

    forcex[pidx] =    ( wijk[pidx]   * fieldx[jx   + ix*stridex     + kx*stridex*stridey]
                      + wij1k[pidx]  * fieldx[jx+1 + ix*stridex     + kx*stridex*stridey]
                      + wi1jk[pidx]  * fieldx[jx+  + (ix+1)*stridex + kx*stridex*stridey]
                      + wi1j1k[pidx] * fieldx[jx+1 + (ix+1)*stridex + kx*stridex*stridey]
                      + wijk1[pidx]  * fieldx[jx   + ix*stridex     + (kx+1)*stridex*stridey]
                      + wij1k1[pidx] * fieldx[jx+1 + ix*stridex     + (kx+1)*stridex*stridey]
                      + wi1jk1[pidx] * fieldx[jx+  + (ix+1)*stridex + (kx+1)*stridex*stridey]
                      + wi1j1k1[pidx]* fieldx[jx+1 + (ix+1)*stridex + (kx+1)*stridex*stridey]);

    forcey[pidx] =    ( wijk[pidx]   * fieldy[jx   + ix*stridex     + kx*stridex*stridey]
                      + wij1k[pidx]  * fieldy[jx+1 + ix*stridex     + kx*stridex*stridey]
                      + wi1jk[pidx]  * fieldy[jx+  + (ix+1)*stridex + kx*stridex*stridey]
                      + wi1j1k[pidx] * fieldy[jx+1 + (ix+1)*stridex + kx*stridex*stridey]
                      + wijk1[pidx]  * fieldy[jx   + ix*stridex     + (kx+1)*stridex*stridey]
                      + wij1k1[pidx] * fieldy[jx+1 + ix*stridex     + (kx+1)*stridex*stridey]
                      + wi1jk1[pidx] * fieldy[jx+  + (ix+1)*stridex + (kx+1)*stridex*stridey]
                      + wi1j1k1[pidx]* fieldy[jx+1 + (ix+1)*stridex + (kx+1)*stridex*stridey]);

    forcez[pidx] =    ( wijk[pidx]   * fieldz[jx   + ix*stridex     + kx*stridex*stridey]
                      + wij1k[pidx]  * fieldz[jx+1 + ix*stridex     + kx*stridex*stridey]
                      + wi1jk[pidx]  * fieldz[jx+  + (ix+1)*stridex + kx*stridex*stridey]
                      + wi1j1k[pidx] * fieldz[jx+1 + (ix+1)*stridex + kx*stridex*stridey]
                      + wijk1[pidx]  * fieldz[jx   + ix*stridex     + (kx+1)*stridex*stridey]
                      + wij1k1[pidx] * fieldz[jx+1 + ix*stridex     + (kx+1)*stridex*stridey]
                      + wi1jk1[pidx] * fieldz[jx+  + (ix+1)*stridex + (kx+1)*stridex*stridey]
                      + wi1j1k1[pidx]* fieldz[jx+1 + (ix+1)*stridex + (kx+1)*stridex*stridey]);

}

__global__ void mesh_to_particles_3d(double* particles_quantity, double *mesh_quantity,
                                     const int stridex, const int stridey,
                                     double *wijk, double *wi1jk, double *wij1k, double *wi1j1k,
                                     double *wijk1, double *wi1jk1, double* wij1k1, double* wi1j1k1,
                                     int *i, int *j, int* k)
{
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    int jj = blockIdx.y*blockDim.y + threadIdx.y;
    int kk = blockIdx.z;
   // int pidx = ii + jj*stridex + kk*stridex*stridey;
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;
    int ix = i[pidx];
    int jx = j[pidx];
    int kx = k[pidx];

    particles_quantity[pidx] = ( wijk[pidx]   * mesh_quantity[jx   + ix*stridex     + kx*stridex*stridey]
                               + wij1k[pidx]  * mesh_quantity[jx+1 + ix*stridex     + kx*stridex*stridey]
                               + wi1jk[pidx]  * mesh_quantity[jx+  + (ix+1)*stridex + kx*stridex*stridey]
                               + wi1j1k[pidx] * mesh_quantity[jx+1 + (ix+1)*stridex + kx*stridex*stridey]
                               + wijk1[pidx]  * mesh_quantity[jx   + ix*stridex     + (kx+1)*stridex*stridey]
                               + wij1k1[pidx] * mesh_quantity[jx+1 + ix*stridex     + (kx+1)*stridex*stridey]
                               + wi1jk1[pidx] * mesh_quantity[jx+  + (ix+1)*stridex + (kx+1)*stridex*stridey]
                               + wi1j1k1[pidx]* mesh_quantity[jx+1 + (ix+1)*stridex + (kx+1)*stridex*stridey]);
}

} /* end extern C */
