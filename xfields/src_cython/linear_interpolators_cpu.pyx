cdef extern from "linear_interpolators_cpu.h" :
    void p2m_rectmesh3d(
        const int nparticles,
        double* x, double* y, double* z,
	double* part_weights,
        const double x0, const double y0, const double z0,
        const double dx, const double dy, const double dz,
        const int nx, const int ny, const int nz,
        double *grid1d
    );

    void m2p_rectmesh3d(
        const int nparticles,
        double* x, double* y, double* z,
        const double x0, const double y0, const double z0,
        const double dx, const double dy, const double dz,
        const int nx, const int ny, const int nz,
        const int n_quantities,
        const int* offsets_mesh_quantities,
        double* mesh_quantity,
        double* particles_quantity
    );


def p2m(double[::1] x, double[::1] y, double[::1] z,
        double[::1] part_weights,
        double x0, double y0, double z0,
        double dx, double dy, double dz,
        int nx, int ny, int nz, double[::1, :, :] rho):

    assert len(x) == len(y) == len(z)

    p2m_rectmesh3d(
	len(x),
        &x[0], &y[0],&z[0],
        &part_weights[0],
        x0, y0, z0,
        dx, dy, dz,
        nx, ny, nz,
        &rho[0,0,0])

from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
def m2p(
        double[::1] x, double[::1] y, double[::1] z,
        double x0, double y0, double z0,
        double dx, double dy, double dz,
        int nx, int ny, int nz,
        int n_maps,
        int[::1] offsets_mesh_quantities,
        double[::1, :, :, :] mesh_quantities,
        double[::1] particles_quantities):


    m2p_rectmesh3d(
         len(x),
         &x[0], &y[0],&z[0],
         x0, y0, z0,
         dx, dy, dz,
         nx, ny, nz,
         n_maps,
         &offsets_mesh_quantities[0],
         &mesh_quantities[0,0,0,0],
         &particles_quantities[0])

