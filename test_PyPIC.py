
from __future__ import division

import unittest
import numpy as np
import matplotlib.pyplot as plt
from pypic import PyPIC
from meshing import RectMesh2D
import poisson_solver.FD_solver as FD
class TestPyPIC(unittest.TestCase):
    def test_pypic_cpu_new(self):
        mesh = RectMesh2D(x0=-0.5, y0=-0.5, dx=0.1, dy=0.1, nx=16, ny=16)
        #mesh = RectMesh2D(x0=0.0, y0=0.0, dx=0.1, dy=0.1, nx=10, ny=10)
        poissonsolver = FD.CPUFiniteDifferencePoissonSolver(mesh,
                laplacian_stencil=FD.laplacian_2D_5stencil)
        pp = PyPIC(mesh, poissonsolver)
        n_particles = 10000
        np.random.seed(0)
        xx = np.random.normal(0.0, 0.1, n_particles)
        yy = np.random.normal(0.0, 0.1, n_particles)
        fx, fy = pp.pic_solve(xx,yy)
        fig = plt.figure()
        f = plt.scatter(xx,yy,s=40,c=fx)
        plt.colorbar()
        plt.show()

    def gen_fake_chamber(self, mesh):
        class fake_chamb():
            def __init__(self, mesh):
                self.mesh = mesh
                self.x_aper = self.mesh.dx*(self.mesh.nx-1)/2.
                self.y_aper = self.mesh.dy*(self.mesh.ny-1)/2.
                self.is_outside = np.vectorize(self.is_outside_scalar)
            def is_outside_scalar(self, x, y):
                #in_x = self.mesh.x0 < x < self.mesh.x0 + self.mesh.dx*(self.mesh.nx-1)
                #in_y = self.mesh.y0 < y < self.mesh.y0 + self.mesh.dy*(self.mesh.ny-1)
                in_x = -self.x_aper < x < self.x_aper
                in_y = -self.y_aper < y < self.y_aper
                return not(in_x and in_y)
        chamb = fake_chamb(mesh)
        return chamb

    def make_centered_mesh(self, Lx, Ly, nx, ny):
        dx = Lx/(nx-1)
        dy = Ly/(ny-1)
        x0 = -Lx/2.
        y0 = -Ly/2.
        mesh = RectMesh2D(x0=x0, y0=y0, dx=dy, dy=dy, nx=nx, ny=ny)
        return mesh

    def test_pypic_FiniteDifferences_Staircase_SquareGrid(self):
        #mesh = RectMesh2D(x0=-0.5, y0=-0.5, dx=0.1, dy=0.1, nx=31, ny=31)
        #mesh = RectMesh2D(x0=0., y0=0., dx=0.04, dy=0.04, nx=21, ny=21)
        #mesh = RectMesh2D(x0=-0.5, y0=-0.5, dx=0.1, dy=0.1, nx=11, ny=11)
        mesh = self.make_centered_mesh(1., 1., 23, 23)
        print mesh.nx
        chamber = self.gen_fake_chamber(mesh)
        print(chamber.x_aper)
        poissonsolver = FD.FiniteDifferences_Staircase_SquareGrid(
                chamb=chamber, Dh=mesh.dx)
        pp = PyPIC(mesh, poissonsolver)
        #print(len(pp.poisson_solve.flag_inside_n))
        n_particles = 10000
        np.random.seed(0)
        xx = np.random.normal(0., 0.1, n_particles)
        yy = np.random.normal(0., 0.1, n_particles)
        plt.figure()
        plt.scatter(xx,yy)
        plt.show()
        fx, fy = pp.pic_solve(xx,yy)
        fig = plt.figure()
        f = plt.scatter(xx,yy,s=40,c=fx)
        plt.colorbar()
        plt.show()



if __name__ == '__main__':
    unittest.main()
