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
    mesh_quantity = np.ravel(mesh_quantity)
    particles_quantity = (mesh_quantity[jp   + stridex*ip     + stridex*stridey*kp    ] * weights[0]
                        + mesh_quantity[jp   + stridex*(ip+1) + stridex*stridey*kp    ] * weights[1]
                        + mesh_quantity[jp+1 + stridex*ip     + stridex*stridey*kp    ] * weights[2]
                        + mesh_quantity[jp+1 + stridex*(ip+1) + stridex*stridey*kp    ] * weights[3]
                        + mesh_quantity[jp   + stridex*ip     + stridex*stridey*(kp+1)] * weights[4]
                        + mesh_quantity[jp   + stridex*(ip+1) + stridex*stridey*(kp+1)] * weights[5]
                        + mesh_quantity[jp+1 + stridex*ip     + stridex*stridey*(kp+1)] * weights[6]
                        + mesh_quantity[jp+1 + stridex*(ip+1) + stridex*stridey*(kp+1)] * weights[7])
    return particles_quantity

def mesh_to_particles_CPU_2d(mesh, mesh_quantity, indices, weights):
    """CPU kernel for 3d mesh to particles quantity interpolation"""
    ip, jp = indices
    def check_outside(ip, jp):
        outside_idx = ip < 0 and jp < 0 and ip >= mesh.ny and jp >= mesh.nx
        return outside_idx
    check_out = np.vectorize(check_outside)
    outside_idx = check_out(ip,jp)

    mesh_quantity = np.ravel(mesh_quantity)
    stridex = mesh.nx
    particles_quantity = (mesh_quantity[jp   + stridex*ip    ] * weights[0]
                        + mesh_quantity[jp   + stridex*(ip+1)] * weights[1]
                        + mesh_quantity[jp+1 + stridex*ip    ] * weights[2]
                        + mesh_quantity[jp+1 + stridex*(ip+1)] * weights[3])
    particles_quantity[outside_idx] = 0
    return particles_quantity
