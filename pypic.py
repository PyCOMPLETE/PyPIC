import numpy as np
from scipy.constants import e
import os


try:
    from pycuda import driver as cuda
    from pycuda import gpuarray
    from pycuda.compiler import SourceModule
    from pycuda.tools import DeviceData
except ImportError:
    print('pycuda not found. no gpu capabilities will be available')

from gradient.gradient import make_GPU_gradient
from m2p.m2p import mesh_to_particles_CPU_2d, mesh_to_particles_CPU_3d
from p2m.p2m import particles_to_mesh_CPU_2d, particles_to_mesh_CPU_3d

class PyPIC_GPU(object):
    '''Encodes the algorithm of PyPIC for a static mesh
    on the GPU:

    - scatter particles to a fixed mesh which yields
      the charge distribution on the mesh
    - solve the discrete Poisson equation on the mesh
      with the charge distribution to obtain the potential
      on the mesh
    - determine electric fields on the mesh from the potential
    - gather the electric fields back to the particles

    Electrostatics are assumed, magnetic fields are neglected.
    Use the Lorentz transformation to determine the
    electric fields in the beam reference frame and then again
    to transform back to the laboratory reference frame,
    hence accounting for the magnetic fields.
    '''
    def __init__(self, mesh, poissonsolver, context):
        '''Mesh sizes need to be powers of 2 in x (and y if it exists).
        '''
        self.mesh = mesh
        self._context = context
        self.poissonsolver = poissonsolver

        # prepare calls to kernels!!!

        # load kernels
        with open('p2m/p2m_kernels.cu') as stream:
            source = stream.read()
        p2m_kernels = SourceModule(source)
        with open('m2p/m2p_kernels.cu') as stream:
            source = stream.read()
        m2p_kernels = SourceModule(source)


        # initialize in init because otherwise it tries to compile even if
        # no instance of the class is created -> errors if you import the module
        # without having a running pycuda context
        # depending on the dimension, the correct funtions are loaded
        self._particles_to_mesh_kernel = (
            p2m_kernels.get_function('particles_to_mesh' +
                                     str(mesh.dimension) + 'd'))
        self._mesh_to_particles_kernel = (
            m2p_kernels.get_function("mesh_to_particles" +
                                     str(mesh.dimension) + 'd'))
        self._field_to_particles_kernel = (
            m2p_kernels.get_function("field_to_particles" +
                                     str(mesh.dimension) + 'd'))

        self._gradient = make_GPU_gradient(mesh, context)


    def particles_to_mesh(self, *mp_coords, **kwargs):
        '''Scatter the macro-particles onto the mesh nodes.
        The argument list mp_coords defines the coordinate arrays of
        the macro-particles, e.g. in 3D
            mp_coords = (x, y, z)
        The keyword argument charge=e is the charge per macro-particle.
        Further keyword arguments are
        mesh_indices=None, mesh_distances=None, mesh_weights=None .

        Return the charge distribution on the mesh.
        '''
        mesh_indices = kwargs.get("mesh_indices",
                                  self.mesh.get_indices(*mp_coords))
        mesh_weights = kwargs.get(
            "mesh_weights", self.mesh.get_weights(
                *mp_coords, indices=mesh_indices,
                distances=kwargs.get("mesh_distances", None)
            )
        )
        charge = kwargs.get("charge", e)
        n_macroparticles = len(mp_coords[0])
        mesh_density = gpuarray.zeros(shape=self.mesh.shape, #self.mesh.n_nodes,
                                      dtype=np.float64)

        self._particles_to_mesh_kernel(
            mesh_density, *(self.mesh.shape[:-1] + mesh_weights + mesh_indices),
            block=(16, 16, 1), grid=(n_macroparticles // 16**2,1,1) # 32x32: too few registers
        )
        self._context.synchronize()
        rho = mesh_density*charge
        return rho

    def poisson_solve(self, rho):
        '''Solve the discrete Poisson equation with the charge
        distribution rho on the mesh, -divgrad phi = rho / epsilon_0 .

        Return the potential phi.
        '''
        # does self._context.synchronize() within solve
        return self.poissonsolver.poisson_solve(rho)
    def poisson_cholsolve(self, rho):
        '''test only'''
        return self.poissonsolver.poisson_cholsolve(rho)

    def get_electric_fields(self, phi):
        '''Return electric fields on the mesh given
        the potential phi on the mesh via
        E = - grad phi .

        Returns asynchronously from the device.
        (You may potentially want to call context.synchronize()!)
        '''
        return self._gradient(-phi)

    def mesh_to_particles(self, mesh_quantity, *mp_coords, **kwargs):
        '''Interpolate the mesh_quantity (whose shape is the mesh shape)
        onto the particles. The argument list mp_coords defines the
        coordinate arrays of the macro-particles, e.g. in 3D
            mp_coords = (x, y, z)
        Possible keyword arguments are
        mesh_indices=None, mesh_distances=None, mesh_weights=None .

        Return the interpolated quantity in an array for each particle.

        Returns asynchronously from the device.
        (You may potentially want to call context.synchronize()!)
        '''
        mesh_indices = kwargs.get("mesh_indices",
                                  self.mesh.get_indices(*mp_coords))
        mesh_weights = kwargs.get(
            "mesh_weights", self.mesh.get_weights(
                *mp_coords, indices=mesh_indices,
                distances=kwargs.get("mesh_distances", None)
            )
        )
        n_macroparticles = len(mp_coords[0])
        particles_quantity = gpuarray.empty(n_macroparticles, dtype=np.float64)

        self._mesh_to_particles_kernel(
            particles_quantity, mesh_quantity,
            *(self.mesh.shape[:-1] + mesh_weights + mesh_indices),
            block=(16, 16, 1), grid=(n_macroparticles // (16*16),1,1)
        )
        return particles_quantity

    def field_to_particles(self, *mesh_fields_and_mp_coords, **kwargs):
        '''Gather the three-dimensional (electric) field
        from the mesh to the particles.
        The list mesh_fields_and_mp_coords consists of 2-tuples for each
        dimension where
        - each first entry is the field array on the mesh,
        - each second entry is the particle coordinate array,
        e.g. in 3D
            mesh_fields_and_mp_coords = ((E_x, x), (E_y, y), (E_z, z))
        where E_x, E_y, E_z would be given by self.get_electric_fields
        and x, y, z are the particle coordinate arrays.
        Possible keyword arguments are
        mesh_indices=None, mesh_distances=None, mesh_weights=None .

        Return the interpolated fields per particle for each dimension.
        '''
        mesh_fields, mp_coords = zip(*mesh_fields_and_mp_coords)
        mesh_indices = kwargs.get("mesh_indices",
                                  self.mesh.get_indices(*mp_coords))
        mesh_weights = kwargs.get(
            "mesh_weights", self.mesh.get_weights(
                *mp_coords, indices=mesh_indices,
                distances=kwargs.get("mesh_distances", None)
            )
        )
        n_macroparticles = len(mp_coords[0])

        # field per particle
        particle_fields = [gpuarray.empty(shape=n_macroparticles,
                                          dtype=np.float64)
                           for _ in mesh_fields]

        args = particle_fields + list(mesh_fields)
        args += self.mesh.shape[:-1] #strides
        args += list(mesh_weights)
        args += list(mesh_indices)
        # interpolate to particles on gpu.
        # interpolation only, multiply with charge afterwards
        self._field_to_particles_kernel(
            *args, block=(16, 16, 1), grid=(n_macroparticles // (16*16),1,1)
        )
        return particle_fields

    def pic_solve(self, *mp_coords, **kwargs):
        '''Encapsulates the whole algorithm to determine the
        fields of the particles on themselves.
        The keyword argument charge=e is the charge per macro-particle.
        Further keyword arguments are
        mesh_indices=None, mesh_distances=None, mesh_weights=None .

        Return as many interpolated fields per particle as
        dimensions in mp_coords are given.
        '''
        mesh_indices = kwargs.get("mesh_indices",
                                  self.mesh.get_indices(*mp_coords))
        mesh_weights = kwargs.get(
            "mesh_weights", self.mesh.get_weights(
                *mp_coords, indices=mesh_indices,
                distances=kwargs.get("mesh_distances", None)
            )
        )
        charge = kwargs.get("charge", e)

        rho = self.particles_to_mesh(*mp_coords, charge=charge,
                                     mesh_indices=mesh_indices,
                                     mesh_weights=mesh_weights)
        phi = self.poisson_solve(rho)
        mesh_e_fields = self.get_electric_fields(phi)
        self._context.synchronize()
        mesh_fields_and_mp_coords = zip(list(mesh_e_fields), list(mp_coords))
        fields = self.field_to_particles(*mesh_fields_and_mp_coords,
                                         mesh_indices=mesh_indices,
                                         mesh_weights=mesh_weights)
        self._context.synchronize()
        return fields

    # PyPIC backwards compatibility
    scatter = particles_to_mesh
    gather = field_to_particles



class PyPIC(object):
    '''Encodes the algorithm of PyPIC for a static mesh
    on the CPU:

    - scatter particles to a fixed mesh which yields
      the charge distribution on the mesh
    - solve the discrete Poisson equation on the mesh
      with the charge distribution to obtain the potential
      on the mesh
    - determine electric fields on the mesh from the potential
    - gather the electric fields back to the particles

    Electrostatics are assumed, magnetic fields are neglected.
    Use the Lorentz transformation to determine the
    electric fields in the beam reference frame and then again
    to transform back to the laboratory reference frame,
    hence accounting for the magnetic fields.
    '''

    def __init__(self, mesh, poissonsolver):
        self.mesh = mesh
        self.poissonsolver = poissonsolver
        if mesh.dimension == 2:
            self._p2m_kernel = particles_to_mesh_CPU_2d
            self._m2p_kernel = mesh_to_particles_CPU_2d
        elif mesh.dimension == 3:
            self._p2m_kernel = particles_to_mesh_CPU_3d
            self._m2p_kernel = mesh_to_particles_CPU_3d
        else:
            raise RuntimeError("Only meshes with dim=2,3 are supported yet")


    def particles_to_mesh(self, *mp_coords, **kwargs):
        '''Scatter the macro-particles onto the mesh nodes.
        The argument list mp_coords defines the coordinate arrays of
        the macro-particles, e.g. in 3D
            mp_coords = (x, y, z)
        The keyword argument charge=e is the charge per macro-particle.
        Further keyword arguments are
        mesh_indices=None, mesh_distances=None, mesh_weights=None .

        Return the charge distribution on the mesh.
        '''
        mesh_indices = kwargs.get("mesh_indices",
                                  self.mesh.get_indices(*mp_coords))
        mesh_weights = kwargs.get(
            "mesh_weights", self.mesh.get_weights(
                *mp_coords, indices=mesh_indices,
                distances=kwargs.get("mesh_distances", None)
            )
        )
        charge = kwargs.get("charge", e)
        n_macroparticles = len(mp_coords[0])
        mesh_density = self._p2m_kernel(self.mesh, n_macroparticles,
                                        mesh_indices, mesh_weights)
        rho = mesh_density*charge
        return rho

    def poisson_solve(self, rho):
        '''Solve the discrete Poisson equation with the charge
        distribution rho on the mesh, -divgrad phi = rho / epsilon_0 .

        Return the potential phi.
        '''
        # does self._context.synchronize() within solve
        return self.poissonsolver.poisson_solve(rho)

    def get_electric_fields(self, phi):
        '''Return electric fields on the mesh given
        the potential phi on the mesh via
        E = - grad phi .

        Returns asynchronously from the device.
        (You may potentially want to call context.synchronize()!)
        '''
        fields = np.gradient(-phi.reshape(list(reversed(self.mesh.shape))))
        return list(reversed(fields))

    def mesh_to_particles(self, mesh_quantity, *mp_coords, **kwargs):
        '''Interpolate the mesh_quantity (whose shape is the mesh shape)
        onto the particles. The argument list mp_coords defines the
        coordinate arrays of the macro-particles, e.g. in 3D
            mp_coords = (x, y, z)
        Possible keyword arguments are
        mesh_indices=None, mesh_distances=None, mesh_weights=None .

        Return the interpolated quantity in an array for each particle.

        Returns asynchronously from the device.
        (You may potentially want to call context.synchronize()!)
        '''
        mesh_indices = kwargs.get("mesh_indices",
                                  self.mesh.get_indices(*mp_coords))
        mesh_weights = kwargs.get(
            "mesh_weights", self.mesh.get_weights(
                *mp_coords, indices=mesh_indices,
                distances=kwargs.get("mesh_distances", None)
            )
        )
        n_macroparticles = len(mp_coords[0])
        particles_quantity = np.empty(n_macroparticles, dtype=np.float64)
        particles_quantity = self._m2p_kernel(self.mesh, mesh_quantity,
                                             mesh_indices, mesh_weights)
        return particles_quantity

    def field_to_particles(self, *mesh_fields_and_mp_coords, **kwargs):
        '''Gather the three-dimensional (electric) field
        from the mesh to the particles.
        The list mesh_fields_and_mp_coords consists of 2-tuples for each
        dimension where
        - each first entry is the field array on the mesh,
        - each second entry is the particle coordinate array,
        e.g. in 3D
            mesh_fields_and_mp_coords = ((E_x, x), (E_y, y), (E_z, z))
        where E_x, E_y, E_z would be given by self.get_electric_fields
        and x, y, z are the particle coordinate arrays.
        Possible keyword arguments are
        mesh_indices=None, mesh_distances=None, mesh_weights=None .

        Return the interpolated fields per particle for each dimension.
        '''
        mesh_fields, mp_coords = zip(*mesh_fields_and_mp_coords)
        mesh_indices = kwargs.get("mesh_indices",
                                  self.mesh.get_indices(*mp_coords))
        mesh_weights = kwargs.get(
            "mesh_weights", self.mesh.get_weights(
                *mp_coords, indices=mesh_indices,
                distances=kwargs.get("mesh_distances", None)
            )
        )
        n_macroparticles = len(mp_coords[0])

        # field per particle
        particle_fields = [np.empty(shape=n_macroparticles,
                                    dtype=np.float64)
                           for _ in mesh_fields]
        for idx, field in enumerate(mesh_fields):
            #call mesh_to_particles once per dimension
            particle_fields[idx] = self.mesh_to_particles(field, *mp_coords,
                                        mesh_indices=mesh_indices,
                                        mesh_weights=mesh_weights)

        return particle_fields

    def pic_solve(self, *mp_coords, **kwargs):
        '''Encapsulates the whole algorithm to determine the
        fields of the particles on themselves.
        The keyword argument charge=e is the charge per macro-particle.
        Further keyword arguments are
        mesh_indices=None, mesh_distances=None, mesh_weights=None .

        Return as many interpolated fields per particle as
        dimensions in mp_coords are given.
        '''
        mesh_indices = kwargs.get("mesh_indices",
                                  self.mesh.get_indices(*mp_coords))
        mesh_weights = kwargs.get(
            "mesh_weights", self.mesh.get_weights(
                *mp_coords, indices=mesh_indices,
                distances=kwargs.get("mesh_distances", None)
            )
        )
        charge = kwargs.get("charge", e)

        rho = self.particles_to_mesh(*mp_coords, charge=charge,
                                     mesh_indices=mesh_indices,
                                     mesh_weights=mesh_weights)
        phi = self.poisson_solve(rho)
        mesh_e_fields = self.get_electric_fields(phi)
        for i, field in enumerate(mesh_e_fields):
            mesh_e_fields[i] = field.flatten()
        mesh_fields_and_mp_coords = zip(list(mesh_e_fields), list(mp_coords))
        fields = self.field_to_particles(*mesh_fields_and_mp_coords,
                                         mesh_indices=mesh_indices,
                                         mesh_weights=mesh_weights)
        return fields

    # PyPIC backwards compatibility
    scatter = particles_to_mesh
    gather = field_to_particles


