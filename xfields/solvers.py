from abc import ABC, abstractmethod


class Solver(ABC):

    @abstractmethod
    def __init__(self, context=None, **kwargs)

    @abstractmethod
    def solve(self, rho):
        return phi

class FFTSolver3D(Solver):

    def solve(self, rho):
        '''
        If an array 3D is passed, it should solve all of them together.
        '''
        pass

class FFTSolver2D(Solver):
    
    def solve(self, rho):
        '''
        If an array 3D is passed, it should solve all of them together.
        '''
        pass

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
