from abc import ABC, abstractmethod


class FieldMap(ABC):

    @abstractmethod
    def __init__(self, context=None, solver=None, solver_type=None,
                 updatable=True, **kwargs):

        '''
        The init will have context argument, specipying the context
        in which wee store the state of the field map
        '''
        if solver is not None:
            'Check conmpatibility with grid'
            self.solver = solver
        elif solver=='generate':
            self.generate_solver(solver_type)

    @abstractmethod
    def generate_solver(self, solver_type):
        pass
        return solver

    # @abstractmethod
    # def get_data_and_singleparticle_code(self):
    #     '''
    #     To be defined, to inject element in
    #     single-particle tracking
    #     '''
    #     pass

    @abstractmethod
    def get_values_at_points(self,
            x, y, z=0,
            return_rho=False,
            return_phi=False,
            return_dphi_dx=False,
            return_dphi_dy=False,
            return_dphi_dz=False):
        pass

    @abstractmethod
    def update_rho(self, rho, reset=True):

        self._assert_updatable()

        self._rho = rho.copy()

    @abstractmethod
    def update_phi(self, phi, reset=True):

        self._assert_updatable()

        self.phi = phi.copy()

    @abstractmethod
    def update_rho_from_particles(x_p, y_p, z_p, ncharges_p, q0, reset=True):
        '''
        If reset is false charge density is added to the stored one
        '''

        self._assert_updatable()

    @abstractmethod
    def update_phi_from_rho(self, solver=None):

        self._assert_updatable()

        if solver is None:
            if hasattr(self, 'solver'):
                solver = self.solver
            else:
                raise ValueError('I have no solver to compute phi!')

    @abstractmethod
    def update_all_from_particles(x_p, y_p, z_p, ncharges_p, q0, reset=True,
                                  solver=None):

        self._assert_updatable()

        self.update_rho_from_particles(
            x_p, y_p, z_p, ncharges_p, q0, reset=reset)

        self.update_phi_from_rho(solver=solver)

    def _assert_updatable(self):
        assert self.updatable, 'This FieldMap is not updatable!'

