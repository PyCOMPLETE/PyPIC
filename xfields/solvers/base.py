from abc import ABC, abstractmethod


class Solver(ABC):

    @abstractmethod
    def __init__(self, context=None, **kwargs):
        pass

    @abstractmethod
    def solve(self, rho):
        return phi


