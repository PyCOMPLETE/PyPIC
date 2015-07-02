from __future__ import division
import numpy as np

from abc import ABCMeta, abstractmethod


def clip_single_element(high, low, val):
    ''' returns low if val < low, high if val > high, else returns val'''
    return low if val <= low else high if val >= high else val

def clip(high, low, val):
    ''' vectorized version of clip'''
    fn = np.vectorize(clip_single_element)
    #return fn(high, low, val)
    return val


class Mesh(object):
    '''Meshes are used for discretising a beam by distributing
    particles onto the mesh nodes. Each mesh node has a unique
    node ID, they are assigned from 0 upwards in steps of 1.
    Quantities such as charge distributions, potentials and
    electric fields etc. may be represented on a Mesh instance's nodes.
    Dimension sizes are given by nx, (ny, nz,) ...
    '''
    __metaclass__ = ABCMeta

    '''Shape of the mesh.'''
    shape = []
    '''Volume element(s) of the mesh.'''
    volume_elem = 0
    '''Distances between nodes in the mesh, list with entries for each
    dimension (each entry may be a list by itself).
    '''
    distances = []
    '''Total number of nodes in this mesh.'''
    n_nodes = 0
    '''Number of boundary nodes in this mesh.'''
    n_boundary_nodes = 0
    '''Math library used for calling math functions
    (e.g. for CPU numpy or for GPU cumath).
    '''
    mathlib = np

    @property
    def dimension(self):
        return len(self.shape)

    @property
    def n_inner_nodes(self):
        return self.n_nodes - self.n_boundary_nodes

    def make_node_iter(self):
        '''Return an iterator iterating over all node IDs in
        ascending order.
        '''
        return xrange(self.n_nodes)

    @abstractmethod
    def decompose_id(self, node_id):
        '''Return decomposition of node_id into indices for
        each dimension.
        '''
        pass

    @abstractmethod
    def is_boundary(self, node_id):
        '''Return boolean whether the given node_id
        lies on the outer boundary of the mesh.
        '''
        pass

    @abstractmethod
    def get_indices(self, *particle_coordinates):
        '''Return indices of particles on mesh.'''
        pass

    @abstractmethod
    def get_node_ids(self, *particle_coordinates):
        '''Return node IDs for particles on mesh.
        A node ID is uniquely determined for each
        of the self.n_nodes nodes and defines an order on them.
        '''
        pass

    @abstractmethod
    def get_distances(self, *particle_coordinates):
        '''Return distances of particles to next mesh node.'''
        pass

    @abstractmethod
    def get_weights(self, *particle_coordinates):
        '''Return weights of mesh nodes surrounding a particle
        when distributing particles onto the mesh nodes.
        '''
        pass

    def boundary_nodes(self):
        '''Return boolean array in order of the node IDs which
        indicates True for each node which is a boundary.
        '''
        vec_is_boundary = np.vectorize(self.is_boundary)
        return vec_is_boundary(self.make_node_iter())

    def get_domain_decomposition(self, max_nodes):
        '''Calculate domain decomposition (tiles) in (x, y, z) for any
        given (1-3 dimensional) mesh taking into account the maximum
        number of nodes, max_nodes.
        Return integer block and grid dimensions for GPU usage when
        launching kernels considering that each mesh node is
        distributed onto one thread.

        Usually max_nodes amounts to the maximum number of threads
        per block. (Use pycuda.tools.DeviceData().max_threads,
        default 1024.) Take into account register use on the kernel
        to optimise for throughput!
        '''
        # use only ints because of pycuda kernel calls which have problems
        # with numpy.int32 .
        nx = getattr(self, 'nx', 1)
        ny = getattr(self, 'ny', 1)
        nz = getattr(self, 'nz', 1)

        if max_nodes % nx:
            raise ValueError('get_domain_composition: max_nodes has to be '
                             'an integer multiple of nx!')
            # one may implement an automatic rotation to optimise and
            # find another dimension that divides max_nodes...
        #threads per block (tpb)
        tpb_x = int(min(nx, max_nodes))
        tpb_y = int(min(ny, max(max_nodes // nx, 1)))
        tpb_z = int(1)

        #blocks per grid (bpg)
        bpg_x = int(max(int(np.ceil(float(nx) / max_nodes)), 1))
        bpg_y = int(ny/tpb_y)
        bpg_z = int(nz)

        grid = (bpg_x, bpg_y, bpg_z)
        block = (tpb_x, tpb_y, tpb_z)
        return block, grid


class RectMesh3D(Mesh):
    '''Rectangular three-dimensional mesh with dimension-wise uniformly
    spaced nodes.
    '''
    def __init__(self, x0, y0, z0, dx, dy, dz, nx, ny, nz, mathlib=np):
        self.mathlib = mathlib
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.distances = (dx, dy, dz)
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.volume_elem = dx*dy*dz
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.nz = np.int32(nz)
        self.shape = (self.nx, self.ny, self.nz)
        self.n_nodes = self.nx * self.ny * self.nz
#         self.n_boundary_nodes = (2*self.nx*self.ny +
#                                  2*(self.nx-1 + self.ny-1) * (self.nz-2))
        self.n_boundary_nodes = self.n_nodes - max(
            0, (self.nx-2)*(self.ny-2)*(self.nz-2))

    def decompose_id(self, node_id):
        '''Return decomposition of node_id into (i, j, k).'''
        # if node_id >= self.n_nodes:
        #     raise IndexError("Given node_id is outside of the range of nodes.")
        j = node_id % self.nx
        i = ((node_id - j) // self.nx) % self.ny
        k = (node_id - j - self.nx*i) // self.ny // self.nx
        return i, j, k

    def is_boundary(self, node_id):
        '''Return boolean whether the given node_id
        lies on the outer boundary of the mesh.
        '''
        i, j, k = self.decompose_id(node_id)
        return (j == 0 or j == self.nx - 1 or
                i == 0 or i == self.ny - 1 or
                k == 0 or k == self.nz - 1)

    def get_indices(self, x, y, z):
        '''Return indices of particles on mesh.
        Calculate indices of each particle on the 3D mesh nodes as
        (i,j,k) where i is down, j is right, k is to the front.
        '''
        j = self.mathlib.floor((x - self.x0)/self.dx).astype(np.int32)
        i = self.mathlib.floor((y - self.y0)/self.dy).astype(np.int32)
        k = self.mathlib.floor((z - self.z0)/self.dz).astype(np.int32)

         # clip to [0, mesh.nx]
        i = clip(self.ny-2, 0, i) # -2: -1 (# cells = # nodes -1) -1 (zero based)
        j = clip(self.nx-2, 0, j)
        k = clip(self.nz-2, 0, k)
        return (i, j, k)

    def get_node_ids(self, x, y, z, indices=None):
        '''Return unique node IDs for particles calculated from
        i, j and k indices. Goes in x from left to right with j,
        then to next line below (in y) with i and after a full 2D block
        in x-y it goes to the next 2D block by one step to the front
        in z with k.
        If indices are given, they are used instead of calling
        get_indices(x, y, z). These indices (i, j, k) may already have
        been determined by a previous call to
        self.get_indices(x, y, z) .
        '''
        # TODO: implement sorting mechanism in addition to node id
        if indices:
            i, j, k = indices
        else:
            i, j, k = self.get_indices(x, y, z)
        return self.nx*self.ny*k + self.nx*i + j

    def get_distances(self, x, y, z, indices=None):
        '''Return distances of particles to next mesh node.
        If indices are given, they are used instead of calling
        get_indices(x, y, z). These indices (i, j, k) may already have
        been determined by a previous call to
        self.get_indices(x, y, z) .
        '''
        if indices:
            i, j, k = indices
        else:
            i, j, k = self.get_indices(x, y, z)
        dx = x - (self.x0 + j*self.dx) #self.dx[i] if dx are not uniform
        dy = y - (self.y0 + i*self.dy)
        dz = z - (self.z0 + k*self.dz)
        return (dx, dy, dz)

    def get_weights(self, x, y, z, distances=None, indices=None):
        '''Return weights of mesh nodes surrounding a particle
        when distributing particles onto the mesh nodes.
        Calculates weights of surrounding nodes in the following order:
            (i,   j,   k  )
            (i+1, j,   k  )
            (i,   j+1, k  )
            (i+1, j+1, k  )
            (i,   j,   k+1)
            (i+1, j,   k+1)
            (i,   j+1, k+1)
            (i+1, j+1, k+1)
        If indices are given, they are used instead of calling
        get_indices(x, y, z). These indices (i, j, k) may already have
        been determined by a previous call to
        get_indices(x, y, z) .
        Alternatively, distances may be given and used instead of
        calling get_distances(x, y, z). Again, the given (dx, dy, dz)
        may come from a previous call to get_distances(x, y, z) .
        '''
        if distances:
            dx, dy, dz = distances
        else:
            dx, dy, dz = self.get_distances(x, y, z, indices)
        weight_ijk =    (1-dx/self.dx)*(1-dy/self.dy)*(1-dz/self.dz)
        weight_i1jk =   (1-dx/self.dx)*(dy/self.dy)  *(1-dz/self.dz)
        weight_ij1k =   (dx/self.dx)  *(1-dy/self.dy)*(1-dz/self.dz)
        weight_i1j1k =  (dx/self.dx)  *(dy/self.dy)  *(1-dz/self.dz)
        weight_ijk1 =   (1-dx/self.dx)*(1-dy/self.dy)*(dz/self.dz)
        weight_i1jk1 =  (1-dx/self.dx)*(dy/self.dy)  *(dz/self.dz)
        weight_ij1k1 =  (dx/self.dx)  *(1-dy/self.dy)*(dz/self.dz)
        weight_i1j1k1 = (dx/self.dx)  *(dy/self.dy)  *(dz/self.dz)
        return (weight_ijk, weight_i1jk, weight_ij1k, weight_i1j1k,
                weight_ijk1, weight_i1jk1, weight_ij1k1, weight_i1j1k1)



class RectMesh2D(Mesh):
    '''Rectangular two-dimensional mesh with dimension-wise uniformly
    spaced nodes.
    '''
    dimension = 2

    def __init__(self, x0, y0, dx, dy, nx, ny, mathlib=np):
        self.mathlib = mathlib
        self.x0 = x0
        self.y0 = y0
        self.distances = (dx, dy)
        self.dx = dx
        self.dy = dy
        self.volume_elem = dx*dy
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.shape = (self.nx, self.ny)
        self.n_nodes = self.nx * self.ny
        self.n_boundary_nodes = self.n_nodes - max(
            0, (self.nx-2)*(self.ny-2))

    def decompose_id(self, node_id):
        '''Return decomposition of node_id into (i, j).'''
        if node_id >= self.n_nodes:
            raise IndexError("Given node_id is outside of the range of nodes.")
        j = node_id % self.nx
        i = ((node_id - j) // self.nx)
        return i, j

    def is_boundary(self, node_id):
        '''Return boolean whether the given node_id
        lies on the outer boundary of the mesh.
        '''
        i, j = self.decompose_id(node_id)
        return (j == 0 or j == self.nx - 1 or
                i == 0 or i == self.ny - 1)

    def get_indices(self, x, y):
        '''Return indices of particles on mesh.
        Calculate indices of each particle on the 2D mesh nodes as
        (i,j) where i is down, j is right.
        '''
        j = self.mathlib.floor((x - self.x0)/self.dx).astype(np.int32)
        i = self.mathlib.floor((y - self.y0)/self.dy).astype(np.int32)

        # clip them to the range [0, mesh.nx]
        j = clip(self.nx-2, 0, j)  # -2: -1 (# cells = # nodes -1) -1 (zero based)
        i = clip(self.ny-2, 0, i)
        return (i, j)

    def get_node_ids(self, x, y, indices=None):
        '''Return unique node IDs for particles calculated from
        i and j indices. Goes in x from left to right with j,
        then to next line below (in y) with i.
        If indices are given, they are used instead of calling
        get_indices(x, y). These indices (i, j) may already have
        been determined by a previous call to
        self.get_indices(x, y).
        '''
        # TODO: implement sorting mechanism in addition to node id
        if indices:
            i, j = indices
        else:
            i, j = self.get_indices(x, y)
        return self.nx*i + j

    def get_distances(self, x, y, indices=None):
        '''Return distances of particles to next mesh node.
        If indices are given, they are used instead of calling
        get_indices(x, y). These indices (i, j) may already have
        been determined by a previous call to
        self.get_indices(x, y).
        '''
        if indices:
            i, j = indices
        else:
            i, j = self.get_indices(x, y)
        dx = x - (self.x0 + j*self.dx) #self.dx[i] if dx are not uniform
        dy = y - (self.y0 + i*self.dy)
        return (dx, dy)

    def get_weights(self, x, y, distances=None, indices=None):
        '''Return weights of mesh nodes surrounding a particle
        when distributing particles onto the mesh nodes.
        Calculates weights of surrounding nodes in the following order:
            (i,   j, )
            (i+1, j, )
            (i,   j+1)
            (i+1, j+1)
        If indices are given, they are used instead of calling
        get_indices(x, y). These indices (i, j) may already have
        been determined by a previous call to
        get_indices(x, y) .
        Alternatively, distances may be given and used instead of
        calling get_distances(x, y). Again, the given (dx, dy)
        may come from a previous call to get_distances(x, y) .
        '''
        if distances:
            dx, dy = distances
        else:
            dx, dy = self.get_distances(x, y, indices)
        weight_ij=   (1-dx/self.dx)*(1-dy/self.dy)
        weight_i1j=  (1-dx/self.dx)*(dy/self.dy)
        weight_ij1=  (dx/self.dx)  *(1-dy/self.dy)
        weight_i1j1= (dx/self.dx)  *(dy/self.dy)
        return (weight_ij, weight_i1j, weight_ij1, weight_i1j1)



class UniformMesh1D(Mesh):
    '''One-dimensional mesh with uniformly spaced nodes.'''
    dimension = 1

    def __init__(self, x0, dx, nx, mathlib=np):
        self.mathlib = mathlib
        self.x0 = x0
        self.distances = (dx,)
        self.dx = dx
        self.volume_elem = dx
        self.nx = np.int32(nx)
        self.shape = (self.nx,)
        self.n_nodes = self.nx
        self.n_boundary_nodes = 2

    def decompose_id(self, node_id):
        '''Return decomposition of node_id into (i,). (Trivial!)'''
        if node_id >= self.n_nodes:
            raise IndexError("Given node_id is outside of the range of nodes.")
        return node_id

    def is_boundary(self, node_id):
        '''Return boolean whether the given node_id
        lies on the outer boundary of the mesh.
        '''
        i, = self.decompose_id(node_id)
        return (i == 0 or i == self.nx - 1)

    def get_indices(self, x):
        '''Return indices of particles on mesh.
        Calculate indices of each particle on the 2D mesh nodes as
        (i,) where i counts to the right.
        '''
        i = self.mathlib.floor((x - self.x0)/self.dx).astype(np.int32)

        # clip them to the range [0, mesh.nx]
        i = clip(self.nx-2, 0, i)  # -2: -1 (# cells = # nodes -1) -1 (zero based)
        return (i,)

    def get_node_ids(self, x, indices=None):
        '''Return unique node IDs for particles calculated from
        the i index. Goes in x from left to right with i.
        If indices are given, they are used instead of calling
        get_indices(x). These indices (i,) may already have
        been determined by a previous call to
        self.get_indices(x).
        '''
        # TODO: implement sorting mechanism in addition to node id
        if indices:
            i, = indices
        else:
            i, = self.get_indices(x)
        return i

    def get_distances(self, x, indices=None):
        '''Return distances of particles to next mesh node.
        If indices are given, they are used instead of calling
        get_indices(x). These indices (i,) may already have
        been determined by a previous call to
        self.get_indices(x).
        '''
        if indices:
            i, = indices
        else:
            i, = self.get_indices(x)
        dx = x - (self.x0 + i*self.dx) #self.dx[i] if dx are not uniform
        return (dx,)

    def get_weights(self, x, distances=None, indices=None):
        '''Return weights of mesh nodes surrounding a particle
        when distributing particles onto the mesh nodes.
        Calculates weights of surrounding nodes in the following order:
            (i,  )
            (i+1,)
        If indices are given, they are used instead of calling
        get_indices(x). These indices (i,) may already have
        been determined by a previous call to
        get_indices(x) .
        Alternatively, distances may be given and used instead of
        calling get_distances(x). Again, the given (dx,)
        may come from a previous call to get_distances(x) .
        '''
        if distances:
            dx, = distances
        else:
            dx, = self.get_distances(x, indices)
        weight_i = 1 - dx/self.dx
        weight_i1 = dx/self.dx
        return (weight_i, weight_i1)
