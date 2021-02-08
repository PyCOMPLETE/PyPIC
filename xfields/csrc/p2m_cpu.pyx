cdef extern from "hellofunctions.h" :
    void p2m_rectmesh3d(
	const int pidx,
        const int nparticles,
        double* x, double* y, double* z,
        const double x0, const double y0, const double z0,
        const double dx, const double dy, const double dz,
        const int nx, const int ny, const int nz,
        double *grid1d
    );


def p2m_1part(int pidx, double[::1] x, double[::1] y, double[::1] z,
        double x0, double y0, double z0,
        double dx, double dy, double dz,
        int nx, int ny, int nz, double[::1] rho):

    p2m_rectmesh3d(
	pidx, len(x),
        &x[0], &y[0],&z[0],
        x0, y0, z0,
        dx, dy, dz,
        nx, ny, nz,
        &rho[0])
