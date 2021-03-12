'''
Abstract base class for poisson solvers
@author Stefan Hegglin, Adrian Oeftiger
'''

from abc import ABCMeta, abstractmethod


class PoissonSolver(object, metaclass=ABCMeta):
    '''PoissonSolver instances are prepared for a fixed parameter set
    (among others a certain mesh). Given a charge distribution rho on this mesh,
    a PoissonSolver solves the corresponding discrete Poisson equation
    for a potential phi:
        -divgrad phi = rho / epsilon_0
    '''
    @abstractmethod
    def poisson_solve(self, rho):
        '''Solve -divgrad phi = rho / epsilon_0 for the potential phi,
        given as input the charge distribution rho on the mesh.
        '''
        pass
