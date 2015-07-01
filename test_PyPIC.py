
from __future__ import division

import unittest
import numpy as np
import matplotlib.pyplot as plt
from pypic import PyPIC, PyPIC_Fortran_M2P_P2M
from meshing import RectMesh2D
import poisson_solver.FD_solver as FD

# chambers
import old.geom_impact_ellip as ell
import old.geom_impact_poly as poly

class TestPyPIC(unittest.TestCase):
    def _test_pypic_cpu_new(self):
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

    def gen_rect_chamber(self, x_aper, y_aper):
        na = np.array
        chamber = poly.polyg_cham_geom_object({'Vx':na([x_aper, -x_aper, -x_aper, x_aper]),
                    'Vy':na([y_aper, y_aper, -y_aper, -y_aper]),
                    'x_sem_ellip_insc':0.99*x_aper,
                    'y_sem_ellip_insc':0.99*y_aper})
        return chamber


    def make_centered_mesh(self, Lx, Ly, nx, ny):
        dx = Lx/(nx-1)
        dy = Ly/(ny-1)
        x0 = -Lx/2.
        y0 = -Ly/2.
        mesh = RectMesh2D(x0=x0, y0=y0, dx=dy, dy=dy, nx=nx, ny=ny)
        return mesh

    def _test_pypic_FiniteDifferences_Staircase_SquareGrid(self):
        #mesh = RectMesh2D(x0=-0.5, y0=-0.5, dx=0.1, dy=0.1, nx=31, ny=31)
        #mesh = RectMesh2D(x0=0., y0=0., dx=0.04, dy=0.04, nx=21, ny=21)
        #mesh = RectMesh2D(x0=-0.5, y0=-0.5, dx=0.1, dy=0.1, nx=11, ny=11)
       # mesh = self.make_centered_mesh(1., 1., 23, 23)
       # print mesh.nx
       # chamber = self.gen_fake_chamber(mesh)
       # print(chamber.x_aper)
       # poissonsolver = FD.FiniteDifferences_Staircase_SquareGrid(
       #         chamb=chamber, Dh=mesh.dx)

        dx = 0.1
        chamber = self.gen_rect_chamber(1, 1)
        poissonsolver = FD.FiniteDifferences_ShortleyWeller_SquareGrid_extrapolation(
                chamb=chamber, Dh=dx)
        mesh = RectMesh2D(poissonsolver.bias_y,
                          poissonsolver.bias_x,
                          dx, dx,
                          poissonsolver.Nyg,
                          poissonsolver.Nxg) #somehow x,y are reversed in the chamber...




        pp = PyPIC(mesh, poissonsolver)
        #print(len(pp.poisson_solve.flag_inside_n))
        n_particles = 10000
        np.random.seed(0)
        xx = np.random.normal(0., 0.1, n_particles)
        yy = np.random.normal(0., 0.1, n_particles)
        plt.figure()
        plt.scatter(xx,yy)
        plt.show()
        [fx, fy] = pp.pic_solve(xx,yy)
        fig = plt.figure()
        f = plt.scatter(xx,yy,s=40,c=fx)
        plt.colorbar()
        plt.show()

    def _test_pypic_FiniteDifferences_ShortleyWeller_SquareGrid(self):
       # mesh = self.make_centered_mesh(2., 2., 23, 23)
       # chamber = self.gen_rect_chamber(1, 1)
       # poissonsolver = FD.FiniteDifferences_ShortleyWeller_SquareGrid(
       #         chamb=chamber, Dh=mesh.dx)

        dx = 0.1
        chamber = self.gen_rect_chamber(1, 1)
        poissonsolver = FD.FiniteDifferences_ShortleyWeller_SquareGrid(
                chamb=chamber, Dh=dx)
        mesh = RectMesh2D(poissonsolver.bias_y,
                          poissonsolver.bias_x,
                          dx, dx,
                          poissonsolver.Nyg,
                          poissonsolver.Nxg)

        pp = PyPIC(mesh, poissonsolver, gradient=poissonsolver.gradient)
        n_particles = 10000
        np.random.seed(0)
        xx = np.random.normal(0., 0.1, n_particles)
        yy = np.random.normal(0., 0.1, n_particles)
        plt.figure()
        plt.scatter(xx,yy)
        plt.show()
        [fx, fy] = pp.pic_solve(xx,yy)
        fig = plt.figure()
        plt.title('ex')
        f = plt.scatter(xx,yy,s=40,c=fx)
        plt.colorbar()
        plt.show()
        fig = plt.figure()
        plt.title('ey')
        f = plt.scatter(xx,yy,s=40,c=fy)
        plt.colorbar()
        plt.show()


    def test_pypic_FiniteDifferences_ShortleyWeller_SquareGrid_extrapolation(self):
        #mesh = self.make_centered_mesh(2., 2., 23, 23)
        dx = 0.1
        chamber = self.gen_rect_chamber(1, 1)
        poissonsolver = FD.FiniteDifferences_ShortleyWeller_SquareGrid_extrapolation(
                chamb=chamber, Dh=dx)
        mesh = RectMesh2D(poissonsolver.bias_y,
                          poissonsolver.bias_x,
                          dx, dx,
                          poissonsolver.Nyg,
                          poissonsolver.Nxg)
        pp = PyPIC(mesh, poissonsolver, gradient=poissonsolver.gradient)
        n_particles = 10000
        np.random.seed(0)
        xx = np.random.normal(0., 0.1, n_particles)
        yy = np.random.normal(0., 0.1, n_particles)
        plt.figure()
        plt.scatter(xx,yy)
        plt.show()
        [fx, fy] = pp.pic_solve(xx,yy)
        fig = plt.figure()
        plt.title('ex')
        f = plt.scatter(xx,yy,s=40,c=fx)
        plt.colorbar()
        plt.show()
        fig = plt.figure()
        plt.title('ey')
        f = plt.scatter(xx,yy,s=40,c=fy)
        plt.colorbar()
        plt.show()

    def _test_FORTRAN_pypic_FiniteDifferences_ShortleyWeller_SquareGrid(self):
        dx = 0.2
        chamber = self.gen_rect_chamber(1, 1)
        poissonsolver = FD.FiniteDifferences_ShortleyWeller_SquareGrid_extrapolation(
                chamb=chamber, Dh=dx)
        mesh = RectMesh2D(poissonsolver.bias_y,
                          poissonsolver.bias_x,
                          dx, dx,
                          poissonsolver.Nyg,
                          poissonsolver.Nxg)
        pp = PyPIC_Fortran_M2P_P2M(mesh, poissonsolver, gradient=poissonsolver.gradient)
        n_particles = 10000
        np.random.seed(0)
        xx = np.random.normal(0., 0.2, n_particles)
        yy = np.random.normal(0., 0.2, n_particles)
        plt.figure()
        plt.scatter(xx,yy)
        plt.show()
        [fx, fy] = pp.pic_solve(xx,yy)
        fig = plt.figure()
        plt.title('ex')
        f = plt.scatter(xx,yy,s=40,c=fx)
        plt.colorbar()
        plt.show()
        fig = plt.figure()
        plt.title('ey')
        f = plt.scatter(xx,yy,s=40,c=fy)
        plt.colorbar()
        plt.show()




if __name__ == '__main__':
    unittest.main()
