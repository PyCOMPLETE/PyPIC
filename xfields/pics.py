from abc import ABC, abstractmethod
from .fieldmaps import (FieldMap, BiGaussianFieldMap, 
	InterpolatedFieldMap, DualGridFieldMap)


class UpdatableFieldMap(FieldMap):

	@update_rho
	def update_rho(self, rho, reset=True):
		pass

	@abstractmethod
	def update_rho_from_particles(x_p, y_p, z_p, ncharges_p, q0, reset=True):
		'''
		If reset is false charge density is added to the stored one
		'''

	def update_phi_from_rho(self, solver=None):

		if solver is None:
			if hasattr(self, 'solver'):
				solver = self.solver
			else:
				raise ValueError('I have no solver to compute phi!')


		

