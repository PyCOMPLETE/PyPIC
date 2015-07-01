'''
Reference implementations (-> slow) for the particles to mesh functions
used in PyPIC
@author Stefan Hegglin, Adrian Oeftiger
'''
import numpy as np

def particles_to_mesh_CPU_3d(mesh, n_macroparticles, mesh_indices, weights):
    """CPU  kernel for 3d mesh to particles interpolation
    Port to Cython
    """
    mesh_density = np.zeros(mesh.n_nodes, dtype=np.float64)
    stridex = mesh.nx
    stridey = mesh.ny
    for p in xrange(n_macroparticles):
        ip = mesh_indices[0][p]
        jp = mesh_indices[1][p]
        kp = mesh_indices[2][p]
        mesh_density [jp   + stridex*ip     + stridex*stridey*kp    ] += weights[0][p]
        mesh_density [jp   + stridex*(ip+1) + stridex*stridey*kp    ] += weights[1][p]
        mesh_density [jp+1 + stridex*ip     + stridex*stridey*kp    ] += weights[2][p]
        mesh_density [jp+1 + stridex*(ip+1) + stridex*stridey*kp    ] += weights[3][p]
        mesh_density [jp   + stridex*ip     + stridex*stridey*(kp+1)] += weights[4][p]
        mesh_density [jp   + stridex*(ip+1) + stridex*stridey*(kp+1)] += weights[5][p]
        mesh_density [jp+1 + stridex*ip     + stridex*stridey*(kp+1)] += weights[6][p]
        mesh_density [jp+1 + stridex*(ip+1) + stridex*stridey*(kp+1)] += weights[7][p]
    return mesh_density


def particles_to_mesh_CPU_2d(mesh, n_macroparticles, mesh_indices, weights):
    """CPU  kernel for 3d mesh to particles interpolation
    """
    mesh_density = np.zeros(mesh.n_nodes, dtype=np.float64)
    stridex = mesh.nx
    for p in xrange(n_macroparticles):
        ip = mesh_indices[0][p]
        jp = mesh_indices[1][p]
        if ip >= 0 and jp >= 0 and ip < mesh.ny and jp < mesh.nx:
            mesh_density [jp   + stridex*ip    ] += weights[0][p]
            mesh_density [jp   + stridex*(ip+1)] += weights[1][p]
            mesh_density [jp+1 + stridex*ip    ] += weights[2][p]
            mesh_density [jp+1 + stridex*(ip+1)] += weights[3][p]
    return mesh_density
