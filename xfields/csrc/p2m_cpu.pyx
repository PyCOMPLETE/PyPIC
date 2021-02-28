cdef extern from "autogen_cpu_p2m.h" :
    void p2m_rectmesh3d(
        const int nparticles,
        double* x, double* y, double* z,
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
        double** mesh_quantity,
        double** particles_quantity
    );


def p2m(double[::1] x, double[::1] y, double[::1] z,
        double x0, double y0, double z0,
        double dx, double dy, double dz,
        int nx, int ny, int nz, double[::1, :, :] rho):

    p2m_rectmesh3d(
	len(x),
        &x[0], &y[0],&z[0],
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
        mesh_quantities,
        particles_quantities):

    assert len(mesh_quantities) == len(particles_quantities)
    cdef int n_maps = len(mesh_quantities)

    cdef double** mq_pointers= \
       <double**>malloc(n_maps * sizeof(double*))
    cdef double** pq_pointers = \
       <double**>malloc(n_maps * sizeof(double*))

    for ii in range(n_maps):
        mq_pointers[ii] = <double*>(
                (<np.ndarray>mesh_quantities[ii]).data)
        pq_pointers[ii] = <double*>(
                (<np.ndarray>particles_quantities[ii]).data)

    m2p_rectmesh3d(
         len(x),
         &x[0], &y[0],&z[0],
         x0, y0, z0,
         dx, dy, dz,
         nx, ny, nz,
         n_maps,
         mq_pointers,
         pq_pointers)

    free(mq_pointers)
    free(pq_pointers)
