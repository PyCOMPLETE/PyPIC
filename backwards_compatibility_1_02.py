'''
Module providing backwards compatibility with PyPIC 1.0.2
@author Stefan Hegglin
'''
import sys
from scipy.constants import e
import numpy as np

from pypic import PyPIC_Fortran_M2P_P2M
from meshing import RectMesh2D
from poisson_solver import FD_solver as FD
from poisson_solver import FFT_solver as FFT


class Proxy_v102(object):
    '''
    Base Class providing the interface of PyPIC 1.0.2 solvers while
    internally using the newer version
    '''
    def __init__(self, pypic):
        '''
        Set up all members which are available in the old PyPIC classes
        '''
        self.pypic = pypic
        self.Dh = pypic.poissonsolver.Dh
        self.xg = self.poissonsolver.xg
        self.yg = self.poissonsolver.yg
        self.bias_x = self.poissonsolver.bias_x
        self.bias_y = self.poissonsolver.bias_y
        self.Nxg = self.poissonsolver.Nxg
        self.Nyg = self.poissonsolver.Nyg
        self.rho = np.zeros((self.Nxg, self.Nyg))
        self.phi = np.zeros((self.Nxg, self.Nyg))
        self.efx = np.zeros((self.Nxg, self.Nyg))
        self.efy = np.zeros((self.Nxg, self.Nyg))

    def scatter_and_solve(self, x_mp, y_mp, nel_mp, charge=-e):
        # all mp must consist of the same number of particles
        assert(nel_mp == nel_mp[0])
        if len(x_mp) > 0:
            mesh_charges = self.pypic.particles_to_mesh(x_mp, y_mp, charge)
            self.rho = mesh_charges.T / self.pypic.mesh.volume_elem
            phi = self.pypic.poisson_solve(mesh_charges)
            self.phi = phi.T
            mesh_e_fields = self.pypic.get_electric_fields(phi)
            self.efx = mesh_e_fields[0].T
            self.efy = mesh_e_fields[1].T
        else:
            self.rho *= 0
            self.phi *= 0
            self.efx *= 0
            self.efy *= 0

    def scatter(self, x_mp, y_mp, nel_mp, charge=-e):
        if len(x_mp) > 0:
            mesh_charges = self.pypic.particles_to_mesh(x_mp, y_mp, charge)
            self.rho = mesh_charges.T / self.pypic.mesh.volume_elem
        else:
            self.rho *= 0

    def solve():
        if len(x_mp) > 0:
            mesh_charges = self.rho.T * self.pypic.mesh.volume_elem
            phi = self.pypic.poisson_solve(mesh_charges)
            self.phi = phi.T
            mesh_e_fields = self.pypic.get_electric_fields(phi)
            self.efx = mesh_e_fields[0].T
            self.efy = mesh_e_fields[1].T
        else:
            self.phi *= 0
            self.efx *= 0
            self.efy *= 0

    def gather(self, x_mp, y_mp):
        if len(x_mp) > 0:
            mesh_e_fields = [self.efx.flatten(), self.efy.flatten()]
            mesh_e_fields_and_mp_coords = zip(list(mesh_e_fields),[x_mp, y_mp])
            E = self.pypic.field_to_particles(*mesh_e_fields_and_mp_coords)
            Ex_sc_n = E[0]
            Ey_sc_n = E[1]
        else:
            Ex_sc_n = 0.
            Ey_sc_n = 0.
        return Ex_sc_n, Ey_sc_n



