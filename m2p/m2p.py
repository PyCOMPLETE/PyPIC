'''
Reference implementations (-> slow) for the mesh to particles functions
used in PyPIC
@author Stefan Hegglin, Adrian Oeftiger
'''
import numpy as np

def mesh_to_particles_CPU_3d(mesh, mesh_quantity, indices, weights):
    """CPU kernel for 3d mesh to particles quantity interpolation"""
    ip, jp, kp = indices
    stridex = mesh.nx
    stridey = mesh.ny
    mq = np.ravel(mesh_quantity)

    @np.vectorize
    def check_outside(ip, jp, kp):
        outside_idx = (jp < 0 or jp >= mesh.nx - 1 or
                       ip < 0 or ip >= mesh.ny - 1 or
                       kp < 0 or kp >= mesh.nz - 1)
        return outside_idx
    outside_idx = check_outside(ip, jp, kp)
    inside_idx = ~outside_idx
    ip, jp, kp = ip[inside_idx], jp[inside_idx], kp[inside_idx]
    weights = map(lambda w: w[inside_idx], weights)

    particles_quantity = np.empty(len(indices[0]), dtype=mesh_quantity.dtype)
    particles_quantity[inside_idx] = (
        mq[jp   + stridex*ip     + stridex*stridey*kp    ] * weights[0]
      + mq[jp   + stridex*(ip+1) + stridex*stridey*kp    ] * weights[1]
      + mq[jp+1 + stridex*ip     + stridex*stridey*kp    ] * weights[2]
      + mq[jp+1 + stridex*(ip+1) + stridex*stridey*kp    ] * weights[3]
      + mq[jp   + stridex*ip     + stridex*stridey*(kp+1)] * weights[4]
      + mq[jp   + stridex*(ip+1) + stridex*stridey*(kp+1)] * weights[5]
      + mq[jp+1 + stridex*ip     + stridex*stridey*(kp+1)] * weights[6]
      + mq[jp+1 + stridex*(ip+1) + stridex*stridey*(kp+1)] * weights[7])

    particles_quantity[outside_idx] = 0
    return particles_quantity

def mesh_to_particles_CPU_2d(mesh, mesh_quantity, indices, weights):
    """CPU kernel for 3d mesh to particles quantity interpolation"""
    ip, jp = indices
    stridex = mesh.nx
    mesh_quantity = np.ravel(mesh_quantity)

    @np.vectorize
    def check_outside(ip, jp):
        outside_idx = (jp < 0 or jp >= mesh.nx - 1 or
                       ip < 0 or ip >= mesh.ny - 1)
        return outside_idx
    outside_idx = check_outside(ip, jp)
    inside_idx = ~outside_idx
    ip, jp = ip[inside_idx], jp[inside_idx]
    weights = map(lambda w: w[inside_idx], weights)

    particles_quantity = np.empty(len(indices[0]), dtype=mesh_quantity.dtype)
    particles_quantity[inside_idx] = (
        mesh_quantity[jp   + stridex*ip    ] * weights[0]
      + mesh_quantity[jp   + stridex*(ip+1)] * weights[1]
      + mesh_quantity[jp+1 + stridex*ip    ] * weights[2]
      + mesh_quantity[jp+1 + stridex*(ip+1)] * weights[3])

    particles_quantity[outside_idx] = 0
    return particles_quantity
