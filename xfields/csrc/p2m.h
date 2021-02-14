
void p2m_rectmesh3d(
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

    double vol_m1 = 1/(dx*dy*dz);

    int pidx = 0; //vectorize_over pidx nparticles

    // indices
    int jx = floor((x[pidx] - x0) / dx);
    int ix = floor((y[pidx] - y0) / dy);
    int kx = floor((z[pidx] - z0) / dz);

    // distances
    double dxi = x[pidx] - (x0 + jx * dx);
    double dyi = y[pidx] - (y0 + ix * dy);
    double dzi = z[pidx] - (z0 + kx * dz);

    // weights
    double wijk =    vol_m1 * (1.-dxi/dx) * (1.-dyi/dy) * (1.-dzi/dz);
    double wi1jk =   vol_m1 * (1.-dxi/dx) * (dyi/dy)    * (1.-dzi/dz);
    double wij1k =   vol_m1 * (dxi/dx)    * (1.-dyi/dy) * (1.-dzi/dz);
    double wi1j1k =  vol_m1 * (dxi/dx)    * (dyi/dy)    * (1.-dzi/dz);
    double wijk1 =   vol_m1 * (1.-dxi/dx) * (1.-dyi/dy) * (dzi/dz);
    double wi1jk1 =  vol_m1 * (1.-dxi/dx) * (dyi/dy)    * (dzi/dz);
    double wij1k1 =  vol_m1 * (dxi/dx)    * (1.-dyi/dy) * (dzi/dz);
    double wi1j1k1 = vol_m1 * (dxi/dx)    * (dyi/dy)    * (dzi/dz);

    if (pidx < nparticles) {
        if (jx >= 0 && jx < nx - 1 && ix >= 0 && ix < ny - 1 && kx >= 0 && kx < nz - 1)
        {
            //atomicAdd(&grid1d[jx   + ix*nx     + kx*nx*ny],     wijk);
            //atomicAdd(&grid1d[jx+1 + ix*nx     + kx*nx*ny],     wij1k);
            //atomicAdd(&grid1d[jx   + (ix+1)*nx + kx*nx*ny],     wi1jk);
            //atomicAdd(&grid1d[jx+1 + (ix+1)*nx + kx*nx*ny],     wi1j1k);
            //atomicAdd(&grid1d[jx   + ix*nx     + (kx+1)*nx*ny], wijk1);
            //atomicAdd(&grid1d[jx+1 + ix*nx     + (kx+1)*nx*ny], wij1k1);
            //atomicAdd(&grid1d[jx   + (ix+1)*nx + (kx+1)*nx*ny], wi1jk1);
            //atomicAdd(&grid1d[jx+1 + (ix+1)*nx + (kx+1)*nx*ny], wi1j1k1);
            grid1d[jx   + ix*nx     + kx*nx*ny] = 
		    grid1d[jx   + ix*nx     + kx*nx*ny] +    wijk;
            grid1d[jx+1 + ix*nx     + kx*nx*ny] = 
		    grid1d[jx+1 + ix*nx     + kx*nx*ny] +    wij1k;
            grid1d[jx   + (ix+1)*nx + kx*nx*ny] = 
		    grid1d[jx   + (ix+1)*nx + kx*nx*ny] +    wi1jk;
            grid1d[jx+1 + (ix+1)*nx + kx*nx*ny] = 
		    grid1d[jx+1 + (ix+1)*nx + kx*nx*ny] +    wi1j1k;
            grid1d[jx   + ix*nx     + (kx+1)*nx*ny] = 
		    grid1d[jx   + ix*nx     + (kx+1)*nx*ny] +wijk1;
            grid1d[jx+1 + ix*nx     + (kx+1)*nx*ny] = 
		    grid1d[jx+1 + ix*nx     + (kx+1)*nx*ny] +wij1k1;
            grid1d[jx   + (ix+1)*nx + (kx+1)*nx*ny] = 
		    grid1d[jx   + (ix+1)*nx + (kx+1)*nx*ny] +wi1jk1;
            grid1d[jx+1 + (ix+1)*nx + (kx+1)*nx*ny] = 
		    grid1d[jx+1 + (ix+1)*nx + (kx+1)*nx*ny] +wi1j1k1;
        }
    }
    //end_vectorize
}


void m2p_rectmesh3d_scalar(
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

    int pidx = 0; //vectorize_over pidx nparticles


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
    //end_vectorize
}
