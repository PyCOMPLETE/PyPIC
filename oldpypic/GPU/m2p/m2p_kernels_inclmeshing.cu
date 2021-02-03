/*
   GPU Kernels for the mesh to particles functions including meshing part
   @author: Adrian Oeftiger
*/

extern "C" {

// RectMesh2D variants

__global__ void m2p_rectmesh2d_scalar(
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
      // scalar field defined over mesh
    double *mesh_quantity,
    // OUTPUTS:
    double* particles_quantity
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
            particles_quantity[pidx] = ( wij   * mesh_quantity[jx   + ix*nx]
                                       + wij1  * mesh_quantity[jx+1 + ix*nx]
                                       + wi1j  * mesh_quantity[jx+  + (ix+1)*nx]
                                       + wi1j1 * mesh_quantity[jx+1 + (ix+1)*nx]);
        } else {
            particles_quantity[pidx] = 0;
        }
    }
}

__global__ void m2p_rectmesh2d_vector(
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
      // field vector components defined over mesh
    double* fieldx, double* fieldy,
    // OUTPUTS:
    double* forcex, double* forcey
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
            forcex[pidx] =    ( wij *  fieldx[jx + ix*nx]
                              + wij1 * fieldx[jx+1 + ix*nx]
                              + wi1j * fieldx[jx + (ix+1)*nx]
                              + wi1j1 *fieldx[jx+1 + (ix+1)*nx]);
            forcey[pidx] =    ( wij *  fieldy[jx + ix*nx]
                              + wij1 * fieldy[jx+1 + ix*nx]
                              + wi1j * fieldy[jx + (ix+1)*nx]
                              + wi1j1 *fieldy[jx+1 + (ix+1)*nx]);
        } else {
            forcex[pidx] = 0;
            forcey[pidx] = 0;
        }
    }
}


// RectMesh3D variants

__global__ void m2p_rectmesh3d_scalar(
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
      // scalar field defined over mesh
    double *mesh_quantity,
    // OUTPUTS:
    double* particles_quantity
) {
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;

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
            particles_quantity[pidx] = ( wijk   * mesh_quantity[jx   + ix*nx     + kx*nx*ny]
                                       + wij1k  * mesh_quantity[jx+1 + ix*nx     + kx*nx*ny]
                                       + wi1jk  * mesh_quantity[jx+  + (ix+1)*nx + kx*nx*ny]
                                       + wi1j1k * mesh_quantity[jx+1 + (ix+1)*nx + kx*nx*ny]
                                       + wijk1  * mesh_quantity[jx   + ix*nx     + (kx+1)*nx*ny]
                                       + wij1k1 * mesh_quantity[jx+1 + ix*nx     + (kx+1)*nx*ny]
                                       + wi1jk1 * mesh_quantity[jx+  + (ix+1)*nx + (kx+1)*nx*ny]
                                       + wi1j1k1* mesh_quantity[jx+1 + (ix+1)*nx + (kx+1)*nx*ny]);
        } else {
            particles_quantity[pidx] = 0;
        }
    }
}

__global__ void m2p_rectmesh3d_vector(
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
      // field vector components defined over mesh
    double* fieldx, double* fieldy, double* fieldz,
    // OUTPUTS:
    double* forcex, double* forcey, double* forcez
) {
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;

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
            forcex[pidx] = ( wijk   * fieldx[jx   + ix*nx     + kx*nx*ny]
                           + wij1k  * fieldx[jx+1 + ix*nx     + kx*nx*ny]
                           + wi1jk  * fieldx[jx+  + (ix+1)*nx + kx*nx*ny]
                           + wi1j1k * fieldx[jx+1 + (ix+1)*nx + kx*nx*ny]
                           + wijk1  * fieldx[jx   + ix*nx     + (kx+1)*nx*ny]
                           + wij1k1 * fieldx[jx+1 + ix*nx     + (kx+1)*nx*ny]
                           + wi1jk1 * fieldx[jx+  + (ix+1)*nx + (kx+1)*nx*ny]
                           + wi1j1k1* fieldx[jx+1 + (ix+1)*nx + (kx+1)*nx*ny]);

            forcey[pidx] = ( wijk   * fieldy[jx   + ix*nx     + kx*nx*ny]
                           + wij1k  * fieldy[jx+1 + ix*nx     + kx*nx*ny]
                           + wi1jk  * fieldy[jx+  + (ix+1)*nx + kx*nx*ny]
                           + wi1j1k * fieldy[jx+1 + (ix+1)*nx + kx*nx*ny]
                           + wijk1  * fieldy[jx   + ix*nx     + (kx+1)*nx*ny]
                           + wij1k1 * fieldy[jx+1 + ix*nx     + (kx+1)*nx*ny]
                           + wi1jk1 * fieldy[jx+  + (ix+1)*nx + (kx+1)*nx*ny]
                           + wi1j1k1* fieldy[jx+1 + (ix+1)*nx + (kx+1)*nx*ny]);

            forcez[pidx] = ( wijk   * fieldz[jx   + ix*nx     + kx*nx*ny]
                           + wij1k  * fieldz[jx+1 + ix*nx     + kx*nx*ny]
                           + wi1jk  * fieldz[jx+  + (ix+1)*nx + kx*nx*ny]
                           + wi1j1k * fieldz[jx+1 + (ix+1)*nx + kx*nx*ny]
                           + wijk1  * fieldz[jx   + ix*nx     + (kx+1)*nx*ny]
                           + wij1k1 * fieldz[jx+1 + ix*nx     + (kx+1)*nx*ny]
                           + wi1jk1 * fieldz[jx+  + (ix+1)*nx + (kx+1)*nx*ny]
                           + wi1j1k1* fieldz[jx+1 + (ix+1)*nx + (kx+1)*nx*ny]);
        } else {
            forcex[pidx] = 0;
            forcey[pidx] = 0;
            forcez[pidx] = 0;
        }
    }
}

} /* end extern C */
