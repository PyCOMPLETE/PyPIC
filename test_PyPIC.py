import unittest
import numpy as np
import matplotlib.pyplot as plt

from pypic import PyPIC
from meshing import RectMesh2D
from poisson_solver.FD_solver import CPUFiniteDifferencePoissonSolver, laplacian_2D_5stencil


if __name__ == '__main__':
    #unittest.main()
    mesh = RectMesh2D(x0=0.0, y0=0.0, dx=0.1, dy=0.1, nx=10, ny=10)
    poissonsolver = CPUFiniteDifferencePoissonSolver(mesh,
            laplacian_stencil=laplacian_2D_5stencil)
    pp = PyPIC(mesh, poissonsolver)
    n_particles = 1000
    xx = np.random.normal(0.5, 0.1, n_particles)
    yy = np.random.normal(0.5, 0.1, n_particles)
    fx, fy = pp.pic_solve(xx,yy)
    fig = plt.figure()
    f = plt.scatter(xx,yy,s=40,c=fx)
    plt.colorbar()
    plt.show()
