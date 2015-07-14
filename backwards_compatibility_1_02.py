'''
Module providing backwards compatibility with PyPIC 1.0.2
How to: Add 'from backwards_compatibility_1_02 import *' to the beginning
        of your old scripts
Main idea: 1) Wrap the new classes with the old interface
           2) Create new modules (via imp) with the same names as the old ones
           3) Add them to sys.modules
@author Stefan Hegglin
@date 12.07.2015
'''
import imp
import sys
from scipy.constants import e, epsilon_0
import numpy as np

from pypic import PyPIC_Fortran_M2P_P2M
from meshing import RectMesh2D
from poisson_solver import FD_solver as FD
from poisson_solver import FFT_solver as FFT


# provide a way to use modules from the parent directory
sys.path.append('../')



class _Proxy_v102(object):
    '''
    Base Class providing the interface of PyPIC 1.0.2 solvers while
    internally using the newer version
    '''
    def __init__(self, poissonsolver):
        '''
        Set up all members which are available in the old PyPIC classes
        '''
        mesh = RectMesh2D(poissonsolver.bias_x,
                          poissonsolver.bias_y,
                          poissonsolver.Dh, poissonsolver.Dh,
                          poissonsolver.Nxg,
                          poissonsolver.Nyg)
        if getattr(poissonsolver, 'gradient', None) != None:
            pypic = PyPIC_Fortran_M2P_P2M(mesh, poissonsolver,
                                          poissonsolver.gradient)
        else:
            pypic = PyPIC_Fortran_M2P_P2M(mesh, poissonsolver)
        self.pypic = pypic
        self.Dh = poissonsolver.Dh
        self.xg = poissonsolver.xg
        self.yg = poissonsolver.yg
        self.bias_x = poissonsolver.bias_x
        self.bias_y = poissonsolver.bias_y
        self.Nxg = poissonsolver.Nxg
        self.Nyg = poissonsolver.Nyg
        self.rho = np.zeros((self.Nxg, self.Nyg))
        self.phi = np.zeros((self.Nxg, self.Nyg))
        self.efx = np.zeros((self.Nxg, self.Nyg))
        self.efy = np.zeros((self.Nxg, self.Nyg))

    def scatter_and_solve(self, x_mp, y_mp, nel_mp, charge=-e):
        # all mp must consist of the same number of particles
        assert((nel_mp == nel_mp[0]).all())
        if len(x_mp) > 0:
            charge *= nel_mp[0] # multiply the charge by the # particles per mp
            mesh_charges = self.pypic.particles_to_mesh(x_mp, y_mp,
                    charge=charge)
            self.rho = mesh_charges.T / self.pypic.mesh.volume_elem
            phi = self.pypic.poisson_solve(mesh_charges)
            self.phi = phi.reshape((self.Nyg, self.Nxg)).T
            mesh_e_fields = self.pypic.get_electric_fields(phi)
            self.efx = mesh_e_fields[0].reshape(self.Nyg, self.Nxg).T
            self.efy = mesh_e_fields[1].reshape(self.Nyg, self.Nxg).T
        else:
            self.rho *= 0
            self.phi *= 0
            self.efx *= 0
            self.efy *= 0

    def scatter(self, x_mp, y_mp, nel_mp, charge=-e):
        assert((nel_mp == nel_mp[0]).all())
        if len(x_mp) > 0:
            charge *= nel_mp[0] # multiply the charge by the # particles per mp
            mesh_charges = self.pypic.particles_to_mesh(x_mp, y_mp,
                    charge=charge)
            self.rho = mesh_charges.T / self.pypic.mesh.volume_elem
        else:
            self.rho *= 0

    def solve(self, rho=None, flag_verbose=False):
        if rho == None:
            rho = self.rho.T
        else:
            rho = rho.T
        mesh_charges = rho * self.pypic.mesh.volume_elem
        phi = self.pypic.poisson_solve(mesh_charges)
        self.phi = phi.reshape((self.Nyg, self.Nxg)).T
        mesh_e_fields = self.pypic.get_electric_fields(phi)
        self.efx = mesh_e_fields[0].reshape(self.Nyg, self.Nxg).T
        self.efy = mesh_e_fields[1].reshape(self.Nyg, self.Nxg).T

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


class _FiniteDifferences_Staircase_SquareGrid(_Proxy_v102):
    '''
    Wrapper for the FiniteDifferences_Staircase_SquareGrid class v1.0.2
    Provides the same functionality and interface
    '''
    def __init__(self, chamb, Dh, sparse_solver='scipy_slu'):
        poissonsolver = FD.FiniteDifferences_Staircase_SquareGrid(
                chamb=chamb, Dh=Dh, sparse_solver=sparse_solver,
                ext_boundary=True)
        super(_FiniteDifferences_Staircase_SquareGrid,
                self).__init__(poissonsolver)


class _FiniteDifferences_ShortleyWeller_SquareGrid(_Proxy_v102):
    '''
    Wrapper for the FiniteDifferences_ShortleyWeller_SquareGrid class v1.0.2
    Provides the same functionality and interface
    '''
    def __init__(self, chamb, Dh, sparse_solver='scipy_slu'):
        poissonsolver = FD.FiniteDifferences_ShortleyWeller_SquareGrid(
                chamb=chamb, Dh=Dh, sparse_solver=sparse_solver,
                ext_boundary=True)
        super(_FiniteDifferences_ShortleyWeller_SquareGrid,
                self).__init__(poissonsolver)


class _FiniteDifferences_ShortleyWeller_SquareGrid_extrapolation(_Proxy_v102):
    '''
    Wrapper for the FiniteDifferences_ShortleyWeller_SquareGrid_extrapolation
    class v1.0.2
    Provides the same functionality and interface
    '''
    def __init__(self, chamb, Dh, sparse_solver='scipy_slu'):
        poissonsolver = FD.FiniteDifferences_ShortleyWeller_SquareGrid_extrapolation(
                chamb=chamb, Dh=Dh, sparse_solver=sparse_solver)
        super(_FiniteDifferences_ShortleyWeller_SquareGrid_extrapolation,
                self).__init__(poissonsolver)


class _FFT_OpenBoundary_SquareGrid(_Proxy_v102):
    '''
    Wrapper for the FFT_OpenBoundary_SquareGrid class v1.0.2
    Provides the same functionality and interface
    '''
    def __init__(self, x_aper, y_aper, Dh, fftlib='pyfftw'):
        poissonsolver = FFT.FFT_OpenBoundary_SquareGrid(x_aper=x_aper,
                                                        y_aper=y_aper,
                                                        Dh=Dh,
                                                        fftlib=fftlib)
        super(_FFT_OpenBoundary_SquareGrid, self).__init__(poissonsolver)
        self.fgreen = poissonsolver.fgreen
        self.fgreentr = poissonsolver.fgreentr
        self.nx = len(self.xg)
        self.ny = len(self.yg)
        self.fft2 = poissonsolver.fft2

class _FFT_PEC_Boundary_SquareGrid(_Proxy_v102):
    '''
    Wrapper for the FFT_PEC_Boundary_SquareGrid class v1.0.2
    Provides the same functionality and interface
    '''
    def __init__(self, x_aper, y_aper, Dh, fftlib='pyfftw'):
        poissonsolver = FFT.FFT_PEC_Boundary_SquareGrid(x_aper=x_aper,
                                                        y_aper=y_aper,
                                                        Dh=Dh,
                                                        fftlib=fftlib)
        super(_FFT_PEC_Boundary_SquareGrid, self).__init__(poissonsolver)

# this is where the magic happens
FFT_OpenBoundary_SquareGrid = imp.new_module('FFT_OpenBoundary_SquareGrid')
FFT_PEC_Boundary_SquareGrid = imp.new_module('FFT_PEC_Boundary_SquareGrid')
FiniteDifferences_ShortleyWeller_SquareGrid = imp.new_module('FiniteDifferences_Shortleyweller_SquareGrid')
FiniteDifferences_ShortleyWeller_SquareGrid_extrapolation = imp.new_module('FiniteDifferences_Shortleyweller_SquareGrid_extrapolation')
FiniteDifferences_Staircase_SquareGrid = imp.new_module('FiniteDifferences_Staircase_SquareGrid')
# bind the classes to the correct names inside the modules
FFT_OpenBoundary_SquareGrid.FFT_OpenBoundary_SquareGrid = _FFT_OpenBoundary_SquareGrid
FFT_PEC_Boundary_SquareGrid.FFT_PEC_Boundary_SquareGrid = _FFT_PEC_Boundary_SquareGrid
FiniteDifferences_ShortleyWeller_SquareGrid.FiniteDifferences_ShortleyWeller_SquareGrid = _FiniteDifferences_ShortleyWeller_SquareGrid
FiniteDifferences_ShortleyWeller_SquareGrid_extrapolation.FiniteDifferences_ShortleyWeller_SquareGrid = _FiniteDifferences_ShortleyWeller_SquareGrid_extrapolation
FiniteDifferences_Staircase_SquareGrid.FiniteDifferences_Staircase_SquareGrid = _FiniteDifferences_Staircase_SquareGrid
# add the modules to sys.modules to make statements like
# from FFT_PEC_Boundary_SquareGrid import FFT_PEC_Boundary_Squaregrid valid
sys.modules['FFT_OpenBoundary_SquareGrid'] = FFT_OpenBoundary_SquareGrid
sys.modules['FFT_PEC_Boundary_SquareGrid'] = FFT_PEC_Boundary_SquareGrid
sys.modules['FiniteDifferences_ShortleyWeller_SquareGrid']=FiniteDifferences_ShortleyWeller_SquareGrid
sys.modules['FiniteDifferences_ShortleyWeller_SquareGrid_extrapolation']=FiniteDifferences_ShortleyWeller_SquareGrid_extrapolation
sys.modules['FiniteDifferences_Staircase_SquareGrid'] = FiniteDifferences_Staircase_SquareGrid

