/*
   GPU Kernels for the mesh to particles functions
   @author: Stefan Hegglin, Adrian Oeftiger
*/

extern "C" {

__global__ void mesh_to_particles_2d(
    int nparticles,
    double* particles_quantity, double *mesh_quantity,
    const int nx, const int ny,
    double *wij, double *wi1j, double *wij1, double *wi1j1,
    int *i, int *j)
{
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;
    int ix = i[pidx];
    int jx = j[pidx];

    if (pidx < nparticles) {
        if (jx >= 0 && jx < nx - 1 && ix >= 0 && ix < ny - 1)
        {
            particles_quantity[pidx] = ( wij[pidx]   * mesh_quantity[jx   + ix*nx]
                                       + wij1[pidx]  * mesh_quantity[jx+1 + ix*nx]
                                       + wi1j[pidx]  * mesh_quantity[jx+  + (ix+1)*nx]
                                       + wi1j1[pidx] * mesh_quantity[jx+1 + (ix+1)*nx]);
        } else {
            particles_quantity[pidx] = 0;
        }
    }
}

__global__ void field_to_particles_2d(
    int nparticles,
    double* forcex, double* forcey, double* fieldx, double* fieldy,
    const int nx, const int ny,
    double *wij, double *wi1j, double *wij1, double *wi1j1, int *i, int *j)
{
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;
    int jx = j[pidx];
    int ix = i[pidx];
    if (pidx < nparticles) {
        if (jx >= 0 && jx < nx - 1 && ix >= 0 && ix < ny - 1)
        {
            forcex[pidx] =    ( wij[pidx] *  fieldx[jx + ix*nx]
                              + wij1[pidx] * fieldx[jx+1 + ix*nx]
                              + wi1j[pidx] * fieldx[jx + (ix+1)*nx]
                              + wi1j1[pidx] *fieldx[jx+1 + (ix+1)*nx]);
            forcey[pidx] =    ( wij[pidx] *  fieldy[jx + ix*nx]
                              + wij1[pidx] * fieldy[jx+1 + ix*nx]
                              + wi1j[pidx] * fieldy[jx + (ix+1)*nx]
                              + wi1j1[pidx] *fieldy[jx+1 + (ix+1)*nx]);
        } else {
            forcex[pidx] = 0;
            forcey[pidx] = 0;
        }
    }
}

__global__ void field_to_particles_3d(
    int nparticles,
    double* forcex, double* forcey, double* forcez,
    double* fieldx, double* fieldy, double* fieldz,
    const int nx, const int ny, const int nz,
    double *wijk, double *wi1jk, double *wij1k, double *wi1j1k,
    double *wijk1, double *wi1jk1, double* wij1k1, double* wi1j1k1,
    int *i, int *j, int* k)
{
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;
    int ix = i[pidx];
    int jx = j[pidx];
    int kx = k[pidx];
    if (pidx < nparticles) {
        if (jx >= 0 && jx < nx - 1 && ix >= 0 && ix < ny - 1 && kx >= 0 && kx < nz - 1)
        {
            forcex[pidx] = ( wijk[pidx]   * fieldx[jx   + ix*nx     + kx*nx*ny]
                           + wij1k[pidx]  * fieldx[jx+1 + ix*nx     + kx*nx*ny]
                           + wi1jk[pidx]  * fieldx[jx+  + (ix+1)*nx + kx*nx*ny]
                           + wi1j1k[pidx] * fieldx[jx+1 + (ix+1)*nx + kx*nx*ny]
                           + wijk1[pidx]  * fieldx[jx   + ix*nx     + (kx+1)*nx*ny]
                           + wij1k1[pidx] * fieldx[jx+1 + ix*nx     + (kx+1)*nx*ny]
                           + wi1jk1[pidx] * fieldx[jx+  + (ix+1)*nx + (kx+1)*nx*ny]
                           + wi1j1k1[pidx]* fieldx[jx+1 + (ix+1)*nx + (kx+1)*nx*ny]);

            forcey[pidx] = ( wijk[pidx]   * fieldy[jx   + ix*nx     + kx*nx*ny]
                           + wij1k[pidx]  * fieldy[jx+1 + ix*nx     + kx*nx*ny]
                           + wi1jk[pidx]  * fieldy[jx+  + (ix+1)*nx + kx*nx*ny]
                           + wi1j1k[pidx] * fieldy[jx+1 + (ix+1)*nx + kx*nx*ny]
                           + wijk1[pidx]  * fieldy[jx   + ix*nx     + (kx+1)*nx*ny]
                           + wij1k1[pidx] * fieldy[jx+1 + ix*nx     + (kx+1)*nx*ny]
                           + wi1jk1[pidx] * fieldy[jx+  + (ix+1)*nx + (kx+1)*nx*ny]
                           + wi1j1k1[pidx]* fieldy[jx+1 + (ix+1)*nx + (kx+1)*nx*ny]);

            forcez[pidx] = ( wijk[pidx]   * fieldz[jx   + ix*nx     + kx*nx*ny]
                           + wij1k[pidx]  * fieldz[jx+1 + ix*nx     + kx*nx*ny]
                           + wi1jk[pidx]  * fieldz[jx+  + (ix+1)*nx + kx*nx*ny]
                           + wi1j1k[pidx] * fieldz[jx+1 + (ix+1)*nx + kx*nx*ny]
                           + wijk1[pidx]  * fieldz[jx   + ix*nx     + (kx+1)*nx*ny]
                           + wij1k1[pidx] * fieldz[jx+1 + ix*nx     + (kx+1)*nx*ny]
                           + wi1jk1[pidx] * fieldz[jx+  + (ix+1)*nx + (kx+1)*nx*ny]
                           + wi1j1k1[pidx]* fieldz[jx+1 + (ix+1)*nx + (kx+1)*nx*ny]);
        } else {
            forcex[pidx] = 0;
            forcey[pidx] = 0;
            forcez[pidx] = 0;
        }
    }
}

__global__ void mesh_to_particles_3d(
    int nparticles,
    double* particles_quantity, double *mesh_quantity,
    const int nx, const int ny, const int nz,
    double *wijk, double *wi1jk, double *wij1k, double *wi1j1k,
    double *wijk1, double *wi1jk1, double* wij1k1, double* wi1j1k1,
    int *i, int *j, int* k)
{
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;
    int ix = i[pidx];
    int jx = j[pidx];
    int kx = k[pidx];
    if (pidx < nparticles) {
        if (jx >= 0 && jx < nx - 1 && ix >= 0 && ix < ny - 1 && kx >= 0 && kx < nz - 1)
        {
            particles_quantity[pidx] = ( wijk[pidx]   * mesh_quantity[jx   + ix*nx     + kx*nx*ny]
                                       + wij1k[pidx]  * mesh_quantity[jx+1 + ix*nx     + kx*nx*ny]
                                       + wi1jk[pidx]  * mesh_quantity[jx+  + (ix+1)*nx + kx*nx*ny]
                                       + wi1j1k[pidx] * mesh_quantity[jx+1 + (ix+1)*nx + kx*nx*ny]
                                       + wijk1[pidx]  * mesh_quantity[jx   + ix*nx     + (kx+1)*nx*ny]
                                       + wij1k1[pidx] * mesh_quantity[jx+1 + ix*nx     + (kx+1)*nx*ny]
                                       + wi1jk1[pidx] * mesh_quantity[jx+  + (ix+1)*nx + (kx+1)*nx*ny]
                                       + wi1j1k1[pidx]* mesh_quantity[jx+1 + (ix+1)*nx + (kx+1)*nx*ny]);
        } else {
            particles_quantity[pidx] = 0;
        }
    }
}

} /* end extern C */
