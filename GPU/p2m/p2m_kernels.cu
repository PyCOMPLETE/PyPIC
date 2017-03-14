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

__global__ void particles_to_mesh_2d_64atomics(
        int nparticles, double *grid1d, const int nx, const int ny,
        double *wij, double *wi1j, double *wij1, double *wi1j1, int *i, int *j)

{
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;
    int ix = i[pidx];
    int jx = j[pidx];
    if (pidx < nparticles) {
        if (jx >= 0 && jx < nx - 1 && ix >= 0 && ix < ny - 1)
        {
            atomicAdd(&grid1d[jx + ix*nx], wij[pidx]);
            atomicAdd(&grid1d[jx+1 + ix*nx], wij1[pidx]);
            atomicAdd(&grid1d[jx + (ix+1)*nx], wi1j[pidx]);
            atomicAdd(&grid1d[jx+1 + (ix+1)*nx], wi1j1[pidx]);
        }
    }
}

__global__ void particles_to_mesh_3d_64atomics(
        int nparticles,
        double *grid1d, const int nx, const int ny, const int nz,
        // particle weights:
        double *wijk, double *wi1jk, double *wij1k, double *wi1j1k,
        double *wijk1, double *wi1jk1, double* wij1k1, double* wi1j1k1,
        // particle 3d cell indices
        int *i, int *j, int* k)
{
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;
    int ix = i[pidx];
    int jx = j[pidx];
    int kx = k[pidx];
    if (pidx < nparticles) {
        if (jx >= 0 && jx < nx - 1 && ix >= 0 && ix < ny - 1 && kx >= 0 && kx < nz - 1)
        {
            atomicAdd(&grid1d[jx   + ix*nx     + kx*nx*ny],     wijk[pidx]);
            atomicAdd(&grid1d[jx+1 + ix*nx     + kx*nx*ny],     wij1k[pidx]);
            atomicAdd(&grid1d[jx   + (ix+1)*nx + kx*nx*ny],     wi1jk[pidx]);
            atomicAdd(&grid1d[jx+1 + (ix+1)*nx + kx*nx*ny],     wi1j1k[pidx]);
            atomicAdd(&grid1d[jx   + ix*nx     + (kx+1)*nx*ny], wijk1[pidx]);
            atomicAdd(&grid1d[jx+1 + ix*nx     + (kx+1)*nx*ny], wij1k1[pidx]);
            atomicAdd(&grid1d[jx   + (ix+1)*nx + (kx+1)*nx*ny], wi1jk1[pidx]);
            atomicAdd(&grid1d[jx+1 + (ix+1)*nx + (kx+1)*nx*ny], wi1j1k1[pidx]);
        }
    }
}

__global__ void particles_to_mesh_2d(
        int nparticles, float *grid1d, const int nx, const int ny,
        float *wij, float *wi1j, float *wij1, float *wi1j1, int *i, int *j)

{
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;
    int ix = i[pidx];
    int jx = j[pidx];
    if (pidx < nparticles) {
        if (jx >= 0 && jx < nx - 1 && ix >= 0 && ix < ny - 1)
        {
            atomicAdd(&grid1d[jx   + ix*nx],     wij[pidx]);
            atomicAdd(&grid1d[jx+1 + ix*nx],     wij1[pidx]);
            atomicAdd(&grid1d[jx   + (ix+1)*nx], wi1j[pidx]);
            atomicAdd(&grid1d[jx+1 + (ix+1)*nx], wi1j1[pidx]);
        }
    }
}

__global__ void particles_to_mesh_3d(
        int nparticles,
        float *grid1d, const int nx, const int ny, const int nz,
        // particle weights:
        float *wijk, float *wi1jk, float *wij1k, float *wi1j1k,
        float *wijk1, float *wi1jk1, float* wij1k1, float* wi1j1k1,
        // particle 3d cell indices
        int *i, int *j, int* k)
{
    int pidx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y* blockDim.x + threadIdx.x;
    int ix = i[pidx];
    int jx = j[pidx];
    int kx = k[pidx];
    if (pidx < nparticles) {
        if (jx >= 0 && jx < nx - 1 && ix >= 0 && ix < ny - 1 && kx >= 0 && kx < nz - 1)
        {
            atomicAdd(&grid1d[jx   + ix*nx     + kx*nx*ny],     wijk[pidx]);
            atomicAdd(&grid1d[jx+1 + ix*nx     + kx*nx*ny],     wij1k[pidx]);
            atomicAdd(&grid1d[jx   + (ix+1)*nx + kx*nx*ny],     wi1jk[pidx]);
            atomicAdd(&grid1d[jx+1 + (ix+1)*nx + kx*nx*ny],     wi1j1k[pidx]);
            atomicAdd(&grid1d[jx   + ix*nx     + (kx+1)*nx*ny], wijk1[pidx]);
            atomicAdd(&grid1d[jx+1 + ix*nx     + (kx+1)*nx*ny], wij1k1[pidx]);
            atomicAdd(&grid1d[jx   + (ix+1)*nx + (kx+1)*nx*ny], wi1jk1[pidx]);
            atomicAdd(&grid1d[jx+1 + (ix+1)*nx + (kx+1)*nx*ny], wi1j1k1[pidx]);
        }
    }
}

__global__ void cic_guard_cell_weights_3d(
        // particle positions sorted by cell ids
        double *x_sorted, double *y_sorted, double *z_sorted,
        // mesh
        double x0, double y0, double z0,
        double dx, double dy, double dz,
        int nx, int ny, int n_nodes,
        int* lower_bounds, int* upper_bounds,
        // output: cumulative mesh charges for guard cells
        double* cumweight_ijk, double* cumweight_i1jk,
        double* cumweight_ij1k, double* cumweight_i1j1k,
        double* cumweight_ijk1, double* cumweight_i1jk1,
        double* cumweight_ij1k1, double* cumweight_i1j1k1)
/**
    Calculate the Cloud-in-Cell weights for all particles within a
    guard cell.

    This node-based algorithm expects particle arrays sorted
    by their node id. For each node, the corresponding guard cell
    is spanned from the nodes spatial indices i, j, k to i+1, j+1, k+1.
    (Therefore, the guard cells at the rear boundary nodes do not
    get any contribution for the guard cell nodes that lie outside of
    the original mesh. E.g. i+1 == ny gets a zero entry for sure!)
    Within a guard cell, all particles are weighted according to
    their distance to the respective bounding node.

    The weights of each particle are summed up per guard cell node and
    written back to the global arrays cumweight_ijk etc.

    The index arrays lower_bounds and upper_bounds
    indicate the start and end indices
    within the sorted particle arrays for each node id. The respective
    node id is identical to the index within lower_bounds and
    upper_bounds.
*/
{
    double l_cumweight_ijk,  l_cumweight_i1jk,  l_cumweight_ij1k,  l_cumweight_i1j1k,
           l_cumweight_ijk1, l_cumweight_i1jk1, l_cumweight_ij1k1, l_cumweight_i1j1k1;
    int i, j, k;
    double x0bydx = x0/dx;
    double y0bydy = y0/dy;
    double z0bydz = z0/dz;
    double dx_rel, dy_rel, dz_rel;
    // grid-stride loop
    for (int nid = blockIdx.x * blockDim.x + threadIdx.x;
         nid < n_nodes;
         nid += blockDim.x * gridDim.x)
    {
        j = nid % nx; //& (nx-1); //
        i = ((nid - j) / nx) % ny; //& (ny-1); //
        k = (nid - j - nx * i) / (nx * ny);

        l_cumweight_ijk = 0.;   l_cumweight_i1jk = 0.; l_cumweight_ij1k = 0.;
        l_cumweight_i1j1k = 0.; l_cumweight_ijk1 = 0.; l_cumweight_i1jk1 = 0.;
        l_cumweight_ij1k1 = 0.; l_cumweight_i1j1k1 = 0.;
        for (int pid = lower_bounds[nid]; pid < upper_bounds[nid]; pid++)
        {
            dx_rel = x_sorted[pid]/dx - x0bydx - j;
            dy_rel = y_sorted[pid]/dy - y0bydy - i;
            dz_rel = z_sorted[pid]/dz - z0bydz - k;

            // locally calculate the weights for all 8 nodes of current guard cell
            l_cumweight_ijk +=    (1-dx_rel)*(1-dy_rel)*(1-dz_rel);
            l_cumweight_i1jk +=   (1-dx_rel)*(dy_rel)  *(1-dz_rel);
            l_cumweight_ij1k +=   (dx_rel)  *(1-dy_rel)*(1-dz_rel);
            l_cumweight_i1j1k +=  (dx_rel)  *(dy_rel)  *(1-dz_rel);
            l_cumweight_ijk1 +=   (1-dx_rel)*(1-dy_rel)*(dz_rel);
            l_cumweight_i1jk1 +=  (1-dx_rel)*(dy_rel)  *(dz_rel);
            l_cumweight_ij1k1 +=  (dx_rel)  *(1-dy_rel)*(dz_rel);
            l_cumweight_i1j1k1 += (dx_rel)  *(dy_rel)  *(dz_rel);
        }
        cumweight_ijk[nid] =   l_cumweight_ijk;   cumweight_i1jk[nid] =   l_cumweight_i1jk;
        cumweight_ij1k[nid] =  l_cumweight_ij1k;  cumweight_i1j1k[nid] =  l_cumweight_i1j1k;
        cumweight_ijk1[nid] =  l_cumweight_ijk1;  cumweight_i1jk1[nid] =  l_cumweight_i1jk1;
        cumweight_ij1k1[nid] = l_cumweight_ij1k1; cumweight_i1j1k1[nid] = l_cumweight_i1j1k1;
    }
}

__global__ void join_guard_cells_3d_old( // excl boundary
        double* cumweight_ijk, double* cumweight_i1jk,
        double* cumweight_ij1k, double* cumweight_i1j1k,
        double* cumweight_ijk1, double* cumweight_i1jk1,
        double* cumweight_ij1k1, double* cumweight_i1j1k1,
        int n_nodes, int nx, int ny, int nz,
        double* mesh_charges)
/**

*/
{
    int i, j, k, ijk, i1jk, ij1k, i1j1k, ijk1, i1jk1, ij1k1, i1j1k1;
    // grid-stride loop
    for (int nid = blockIdx.x * blockDim.x + threadIdx.x;
         nid < n_nodes;
         nid += blockDim.x * gridDim.x)
    {
        j = nid % nx; //& (nx-1); //
        i = ((nid - j) / nx) % ny; //& (ny-1); //
        k = (nid - j - nx * i) / (nx * ny);
        if (j == 0 || j == nx - 1 || i == 0 || i == ny - 1 || k == 0 || k == nz - 1)
        {
            continue;
        }

        ijk = nid;                     // nx*ny*k     + nx*i     + j
        i1jk = nid - nx;               // nx*ny*k     + nx*(i-1) + j
        ij1k = nid - 1;                // nx*ny*k     + nx*i     + j-1
        i1j1k = nid - nx - 1;          // nx*ny*k     + nx*(i-1) + j-1
        ijk1 = nid - nx*ny;            // nx*ny*(k-1) + nx*i     + j
        i1jk1 = nid - nx*ny - nx;      // nx*ny*(k-1) + nx*(i-1) + j
        ij1k1 = nid - nx*ny - 1;       // nx*ny*(k-1) + nx*i     + j-1
        i1j1k1 = nid - nx*ny - nx - 1; // nx*ny*(k-1) + nx*(i-1) + j-1


        mesh_charges[nid] =   cumweight_ijk[ijk]     + cumweight_i1jk[i1jk]
                            + cumweight_ij1k[ij1k]   + cumweight_i1j1k[i1j1k]
                            + cumweight_ijk1[ijk1]   + cumweight_i1jk1[i1jk1]
                            + cumweight_ij1k1[ij1k1] + cumweight_i1j1k1[i1j1k1];

    }
}

__global__ void join_guard_cells_3d( // _incl_boundary
        double* cumweight_ijk, double* cumweight_i1jk,
        double* cumweight_ij1k, double* cumweight_i1j1k,
        double* cumweight_ijk1, double* cumweight_i1jk1,
        double* cumweight_ij1k1, double* cumweight_i1j1k1,
        int n_nodes, int nx, int ny, int nz,
        double* mesh_charges)
/**

*/
{
    double l_mesh_charges;
    int j, i, k;
    // grid-stride loop
    for (int nid = blockIdx.x * blockDim.x + threadIdx.x;
         nid < n_nodes;
         nid += blockDim.x * gridDim.x)
    {
        j = nid % nx; //& (nx-1); //
        i = ((nid - j) / nx) % ny; //& (ny-1); //
        k = (nid - j - nx * i) / (nx * ny);

        l_mesh_charges = cumweight_ijk[nid];
        if (j > 0)
            l_mesh_charges += cumweight_ij1k[nid - 1];
        if (i > 0)
            l_mesh_charges += cumweight_i1jk[nid - nx];
        if (j > 0 && i > 0)
            l_mesh_charges += cumweight_i1j1k[nid - nx - 1];
        if (k > 0)
            l_mesh_charges += cumweight_ijk1[nid - nx*ny];
        if (j > 0 && k > 0)
            l_mesh_charges += cumweight_ij1k1[nid - nx*ny - 1];
        if (i > 0 && k > 0)
            l_mesh_charges += cumweight_i1jk1[nid - nx*ny - nx];
        if (j > 0 && i > 0 && k > 0)
            l_mesh_charges += cumweight_i1j1k1[nid - nx*ny - nx - 1];

        mesh_charges[nid] = l_mesh_charges;
    }
}


__global__ void cic_guard_cell_weights_2d(
        // particle positions sorted by cell ids
        double *x_sorted, double *y_sorted,
        // mesh
        double x0, double y0,
        double dx, double dy,
        int nx, int n_nodes,
        int* lower_bounds, int* upper_bounds,
        // output: cumulative mesh charges for guard cells
        double* cumweight_ij, double* cumweight_i1j,
        double* cumweight_ij1, double* cumweight_i1j1)
/**
    Calculate the Cloud-in-Cell weights for all particles within a
    guard cell.

    This node-based algorithm expects particle arrays sorted
    by their node id. For each node, the corresponding guard cell
    is spanned from the nodes spatial indices i, j to i+1, j+1,
    (Therefore, the guard cells at the rear boundary nodes do not
    get any contribution for the guard cell nodes that lie outside of
    the original mesh. E.g. i+1 == ny gets a zero entry for sure!)
    Within a guard cell, all particles are weighted according to
    their distance to the respective bounding node.

    The weights of each particle are summed up per guard cell node and
    written back to the global arrays cumweight_ij etc.

    The index arrays lower_bounds and upper_bounds
    indicate the start and end indices
    within the sorted particle arrays for each node id. The respective
    node id is identical to the index within lower_bounds and
    upper_bounds.
*/
{
    double l_cumweight_ij,  l_cumweight_i1j,  l_cumweight_ij1,  l_cumweight_i1j1;
    int i, j;
    double x0bydx = x0/dx;
    double y0bydy = y0/dy;
    double dx_rel, dy_rel;
    // grid-stride loop
    for (int nid = blockIdx.x * blockDim.x + threadIdx.x;
         nid < n_nodes;
         nid += blockDim.x * gridDim.x)
    {
        j = nid % nx; //& (nx-1); //
        i = ((nid - j) / nx); //& (ny-1); //

        l_cumweight_ij = 0.;   l_cumweight_i1j = 0.; l_cumweight_ij1 = 0.;
        l_cumweight_i1j1 = 0.;
        for (int pid = lower_bounds[nid]; pid < upper_bounds[nid]; pid++)
        {
            dx_rel = x_sorted[pid]/dx - x0bydx - j;
            dy_rel = y_sorted[pid]/dy - y0bydy - i;

            // locally calculate the weights for all 8 nodes of current guard cell
            l_cumweight_ij +=    (1-dx_rel)*(1-dy_rel);
            l_cumweight_i1j +=   (1-dx_rel)*(dy_rel)  ;
            l_cumweight_ij1 +=   (dx_rel)  *(1-dy_rel);
            l_cumweight_i1j1 +=  (dx_rel)  *(dy_rel)  ;
        }
        cumweight_ij[nid] =   l_cumweight_ij;   cumweight_i1j[nid] =   l_cumweight_i1j;
        cumweight_ij1[nid] =  l_cumweight_ij1;  cumweight_i1j1[nid] =  l_cumweight_i1j1;
    }
}

__global__ void join_guard_cells_2d(
        double* cumweight_ij, double* cumweight_i1j,
        double* cumweight_ij1, double* cumweight_i1j1,
        int n_nodes, int nx, int ny,
        double* mesh_charges)
/**

*/
{
    int i, j, ij, i1j, ij1, i1j1;
    // grid-stride loop
    for (int nid = blockIdx.x * blockDim.x + threadIdx.x;
         nid < n_nodes;
         nid += blockDim.x * gridDim.x)
    {
        j = nid % nx; //& (nx-1); //
        i = ((nid - j) / nx); //& (ny-1); //
        if (j == 0 || j == nx - 1 || i == 0 || i == ny - 1)
        {
            continue;
        }

        ij = nid;                     // nx*i     + j
        i1j = nid - nx;               // nx*(i-1) + j
        ij1 = nid - 1;                // nx*i     + j-1
        i1j1 = nid - nx - 1;          // nx*(i-1) + j-1


        mesh_charges[nid] =   cumweight_ij[ij]     + cumweight_i1j[i1j]
                            + cumweight_ij1[ij1]   + cumweight_i1j1[i1j1];

    }
}


} /* end extern C */
