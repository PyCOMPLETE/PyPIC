from .base import Solver

class FDSolver(Solver):

    def __init__(self, boundary_treatment='Shortley-Weller', **kwargs):
        '''
        Boundary treatment can be:
            'Shortley-Weller'
            'Staircase'
        '''
        pass

    def solve(self, rho):
        '''
        If an array 3D is passed, it should solve all of them together.
        '''
