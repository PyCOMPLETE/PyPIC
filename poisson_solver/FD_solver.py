'''
Finite Difference Poisson solvers for PyPIC
@author Stefan Hegglin, Adrian Oeftiger, Giovanni Iadarola
'''

from __future__ import division

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl
from scipy.constants import epsilon_0

from poisson_solver import PoissonSolver

try:
    import PyKLU.klu as klu
except ImportError:
    print('PyKLU not found')
try:
    from pycuda.compiler import SourceModule
    from pycuda import gpuarray
    import libs.cusolver_Rf as curf
except ImportError:
    print('GPU libraries (pycuda, cusolver_RF) not found. GPU functionality ' +
          'not available.')

def invert_permutation(p):
    """Returns an array pinv which corresponds to the inverse permutation
    matrix described by the permutation array p
    """
    m = len(p)
    pinv = np.zeros(m, dtype=np.int32)
    idx = np.arange(m, dtype=np.int32)
    np.put(pinv, p, idx)
    return pinv


def laplacian_3D_7stencil(mesh):
    '''Give first-order nearest-neighbour seven-stencil
    for the discrete 3D negative Laplacian (-divgrad).
    Return list of non-zero matrix row entries in tuples
    with the first entry being the relative position to the diagonal
    and the second entry being the value.
    '''
    stride_x = mesh.nx
    stride_y = mesh.ny
    return ((-stride_x*stride_y, -1),
            (-stride_x, -1),
            (-1, -1),
            (0, 6),
            (1, -1),
            (stride_x, -1),
            (stride_x*stride_y, -1)
    )

def laplacian_2D_5stencil(mesh):
    '''Give first-order nearest-neighbour five-stencil
    for the discrete 2D negative Laplacian (-divgrad).
    Return list of non-zero matrix row entries in tuples
    with the first entry being the relative position to the diagonal
    and the second entry being the value.
    '''
    stride_x = mesh.nx
    return ((-stride_x, -1),
            (-1, -1),
            (0, 4),
            (1, -1),
            (stride_x, -1)
    )


class GPUFiniteDifferencePoissonSolver(PoissonSolver):
    '''Finite difference PoissonSolver on the GPU.
    Uses Dirichlet boundary conditions on a rectangle.
    '''


    def __init__(self, mesh, context, laplacian_stencil=laplacian_3D_7stencil,
                 symmetrize=False, permc_spec='MMD_AT_PLUS_A'):
        '''Assumes that the mesh can accommodate all particles
        on its nodes. Dirichlet boundaries will overwrite all boundary
        nodes for the charge density rho with 0!
        '''
        self.symmetrize = symmetrize
        self.mesh = mesh
        self._context = context
        self._cusolver_handle = curf.cusolverRfCreate()

        self.mesh_inner = gpuarray.to_gpu(~mesh.boundary_nodes())

        lapl_data, lapl_rows, lapl_cols = self.assemble_laplacian(
            mesh, laplacian_stencil)
        A = sps.coo_matrix((lapl_data, (lapl_rows, lapl_cols)),
                           shape=(mesh.n_nodes, mesh.n_nodes),
                           dtype=np.float64
                          ).tocsc()
        if symmetrize:
            """Remove all trivial equations (rows of the matrix where the
            only non-zero element is a 1 on the diagonal. This symmetrizes
            the matrix"""
            N_full = A.shape[0]
            diagonal = A.diagonal()
            indices_non_boundary = np.where(diagonal != 1.)[0]
            N_sel = len(indices_non_boundary)
            Msel = sps.lil_matrix((N_full, N_sel))
            for ii, ind in enumerate(indices_non_boundary):
                Msel[ind,ii] = 1.
            Msel = Msel.tocsc()
            A = (Msel.T*A*Msel).tocsc()
            self.Msel = Msel
            self.MselT = Msel.T
        lu_obj = spl.splu(A, permc_spec=permc_spec)

        L = lu_obj.L.tocsr()
        U = lu_obj.U.tocsr()
        P = lu_obj.perm_r
        Q = lu_obj.perm_c

        # the permutation array definitions required by cusolverRf
        # don't match the ones returnd by spl.splu
        # P <=> inv(P)
        # Q <=> inv(Q)
        P = invert_permutation(P)
        Q = invert_permutation(Q)
        A = A.tocsr()

        self.d_P = gpuarray.to_gpu(P.astype(np.int32))
        self.d_Q = gpuarray.to_gpu(Q.astype(np.int32))
        n = A.shape[0]

        curf.cusolverRfSetupHost(n,
                                 A.nnz, A.indptr, A.indices, A.data,
                                 L.nnz, L.indptr, L.indices, L.data,
                                 U.nnz, U.indptr, U.indices, U.data,
                                 P, Q, self._cusolver_handle)
        self._context.synchronize()

        curf.cusolverRfAnalyze(self._cusolver_handle)
        self._context.synchronize()

        curf.cusolverRfRefactor(self._cusolver_handle)
        self._context.synchronize()

        self.nrhs = 1
        self.ldt = self.mesh.n_nodes
        self.temp = gpuarray.zeros(self.mesh.n_nodes * self.nrhs,
                                   dtype=np.float64)
        self.ldxf = self.mesh.n_nodes

        # determine self.m_sel which removes boundary nodes from laplacian and b
        # put m_sel * LU * m_sel.T onto GPU

        # for testing purposes, cuSOLVER QR factorisation:
        self.lapl_csrVal = gpuarray.to_gpu(A.data)
        self.lapl_csrColInd = gpuarray.to_gpu(A.indices)
        self.lapl_csrRowPtr = gpuarray.to_gpu(A.indptr)
        self._context.synchronize()

    def __del__(self):
        curf.cusolverRfDestroy(self._cusolver_handle)

    @staticmethod
    def assemble_laplacian(mesh, stencil_function):
        '''Assemble the negative Laplacian matrix.
        Return the COO formatted sparse matrix entries as
        tuple of arrays in the order (data, rows, columns)
        '''
        stencil = stencil_function(mesh)
        len_stencil = len(stencil)
        len_data = len_stencil * mesh.n_inner_nodes + mesh.n_boundary_nodes

        data = np.empty(len_data, dtype=np.float64)
        rows = np.empty(len_data, dtype=np.int32)
        cols = np.empty(len_data, dtype=np.int32)
        counter = 0

        # change this to exterior product to allow for non-uniform cell volumes:
        inv_volume_elem = 1. #/ mesh.volume_elem

        for node_id in mesh.make_node_iter():
            if mesh.is_boundary(node_id):
                data[counter] = inv_volume_elem # * 1.
                rows[counter] = node_id
                cols[counter] = node_id
                counter += 1
            else:
                for rel_pos, value in stencil:
                    # inv_volume_elem_at_rel_pos = inv_volume_elem[node_id + rel_pos]
                    data[counter] = value * inv_volume_elem
                    rows[counter] = node_id
                    cols[counter] = node_id + rel_pos
                    global gdata, grows, gcols
                    gdata = data
                    grows = rows
                    gcols = cols
                    counter += 1
        return (data, rows, cols)

    def assemble_rhs(self, rho):
        '''Assemble the right hand side of the Poisson equation,
        -divgrad phi = rho / epsilon_0
        '''
        inv_eps = 1. / epsilon_0
        b = self.mesh_inner * rho * inv_eps
        return b

    def poisson_solve(self, rho):
        '''Return potential phi calculated with LU factorisation.'''
        # b = self.m_sel * rho * self.inv_eps
        b = self.assemble_rhs(rho)
        self._context.synchronize()

        if self.symmetrize:
            # not optimized, simply to check correctness
            bh = b.get()
            bh = self.MselT*bh
            b = gpuarray.to_gpu(bh)

        curf.cusolverRfSolve(self.d_P, self.d_Q, self.nrhs, self.temp,
                             self.ldt, b, self.ldxf, self._cusolver_handle)
        self._context.synchronize()
        if self.symmetrize:
            # not optimized, simply to check correctness
            bh = b.get()
            bh = self.Msel*bh
            b = gpuarray.to_gpu(bh)
        return b


class CPUFiniteDifferencePoissonSolver(PoissonSolver):
    """Finite difference Poisson solver on the CPU.
    Dirichlet boundary conditions on a rectangular grid
    """

    def __init__(self, mesh, laplacian_stencil=laplacian_3D_7stencil):
        self.mesh = mesh
        self.mesh_inner = ~mesh.boundary_nodes()
        lapl_data, lapl_rows, lapl_cols = self.assemble_laplacian(mesh,
                laplacian_stencil) #COO
        A = sps.coo_matrix((lapl_data, (lapl_rows, lapl_cols)),
                           shape=(mesh.n_nodes, mesh.n_nodes),
                           dtype=np.float64).tocsc()
        self.lu_obj = spl.splu(A, permc_spec="MMD_AT_PLUS_A")

    @staticmethod
    def assemble_laplacian(mesh, stencil_function=laplacian_3D_7stencil):
       return GPUFiniteDifferencePoissonSolver.assemble_laplacian(mesh, stencil_function)

    def assemble_rhs(self, rho):
        """Assemble the rhs of the Poisson equation,
        -divgrad phi = rho/epsilon0
        """
        inv_eps = 1. / epsilon_0
        b = self.mesh_inner * rho * inv_eps
        return b

    def poisson_solve(self, rho):
        """ Return the potential (Phi)"""
        b = self.assemble_rhs(rho)
        return self.lu_obj.solve(b)


########## CPU/GPU MIX
class GPUCPUFiniteDifferencePoissonSolver(PoissonSolver):
    '''Finite difference PoissonSolver on the GPU.
    Uses Dirichlet boundary conditions on a rectangle.
    '''
    def __init__(self, mesh, context, laplacian_stencil=laplacian_3D_7stencil,
                 symmetrize=True):
        '''Assumes that the mesh can accommodate all particles
        on its nodes. Dirichlet boundaries will overwrite all boundary
        nodes for the charge density rho with 0!
        '''
        self.symmetrize = symmetrize
        self.mesh = mesh
        self._context = context
        self._cusolver_handle = curf.cusolverRfCreate()

        self.mesh_inner = gpuarray.to_gpu(~mesh.boundary_nodes())

        lapl_data, lapl_rows, lapl_cols = self.assemble_laplacian(
            mesh, laplacian_stencil)
        A = sps.coo_matrix((lapl_data, (lapl_rows, lapl_cols)),
                           shape=(mesh.n_nodes, mesh.n_nodes),
                           dtype=np.float64
                          ).tocsc()
        if symmetrize:
            """Remove all trivial equations (rows of the matrix where the
            only non-zero element is a 1 on the diagonal. This symmetrizes
            the matrix"""
            N_full = A.shape[0]
            diagonal = A.diagonal()
            indices_non_boundary = np.where(diagonal != 1.)[0]
            N_sel = len(indices_non_boundary)
            Msel = sps.lil_matrix((N_full, N_sel))
            for ii, ind in enumerate(indices_non_boundary):
                Msel[ind,ii] = 1.
            Msel = Msel.tocsc()
            A = (Msel.T*A*Msel).tocsc()
            self.Msel = Msel
            self.MselT = Msel.T
        self.lu_obj = klu.Klu(A.tocsc())

    @staticmethod
    def assemble_laplacian(mesh, stencil_function):
        '''Assemble the negative Laplacian matrix.
        Return the COO formatted sparse matrix entries as
        tuple of arrays in the order (data, rows, columns)
        '''
        return GPUFiniteDifferencePoissonSolver.assemble_laplacian(mesh, stencil_function)

    def assemble_rhs(self, rho):
        '''Assemble the right hand side of the Poisson equation,
        -divgrad phi = rho / epsilon_0
        '''
        inv_eps = 1. / epsilon_0
        b = self.mesh_inner * rho * inv_eps
        return b

    def poisson_solve(self, rho):
        '''Return potential phi calculated with LU factorisation.
        compute b on gpu, move to CPU, solve, move back'''
        # b = self.m_sel * rho * self.inv_eps
        b = self.assemble_rhs(rho)
        self._context.synchronize()

        if self.symmetrize:
            # not optimized, simply to check correctness
            bh = b.get()
            bh = self.MselT*bh

        bh = self.lu_obj.solve(bh)

        if self.symmetrize:
            # not optimized, simply to check correctness
            bh = self.Msel*bh
            b = gpuarray.to_gpu(bh)
        return b

###############################################################################
# code below adapted from PyPIC v1.0.2, @author Giovanni Iadarola
###############################################################################

def compute_new_mesh_properties(x_aper=None, y_aper=None, Dh=None, xg=None,
                                yg=None):
    '''Function which returns (Dh, xg, Nxg, bias_x, yg, Nyg, bias_y)
    Guarantees backwards compatibility
    '''
    #TODO put this into the PyPIC class, the Solver should use the grid it
    # gets and not change it!
    if xg!=None and xg!=None:
        assert(x_aper==None and y_aper==None and Dh==None)
        Nxg=len(xg);
        bias_x=min(xg);
        Nyg=len(yg);
        bias_y=min(yg);
        Dh = xg[1]-xg[0]
    else:
        assert(xg==None and xg==None)
        #xg=np.arange(0, x_aper+5.*Dh,Dh,float)
        xg=np.arange(0, x_aper+0.01*Dh,Dh,float)
        xgr=xg[1:]
        xgr=xgr[::-1]#reverse array
        xg=np.concatenate((-xgr,xg),0)
        Nxg=len(xg);
        bias_x=min(xg);
        #yg=np.arange(0,y_aper+4.*Dh,Dh,float)
        yg=np.arange(0,y_aper+0.01*Dh,Dh,float)
        ygr=yg[1:]
        ygr=ygr[::-1]#reverse array
        yg=np.concatenate((-ygr,yg),0)
        Nyg=len(yg);
        bias_y=min(yg);
    return Dh, xg, Nxg, bias_x, yg, Nyg, bias_y

class FiniteDifferences_Staircase_SquareGrid(PoissonSolver):
    '''Finite difference solver using KLU on a square grid. (2d only)
    functionality as class in the old PyPIC with the same name
    '''
    def __init__(self, chamb, Dh, sparse_solver='scipy_slu'):
        # mimics the super() call in the old version
        params = compute_new_mesh_properties(
                chamb.x_aper, chamb.y_aper, Dh)
        print('params: ' + str(params))
        self.Dh, self.xg, self.Nxg, self.bias_x, self.yg, self.Nyg, self.bias_y = params

        [xn, yn]=np.meshgrid(self.xg,self.yg)
        xn=xn.T
        xn=xn.flatten()
        self.xn = xn

        yn=yn.T
        yn=yn.flatten()
        self.yn = yn
        #% xn and yn are stored such that the external index is on x 

        self.flag_outside_n=chamb.is_outside(xn,yn)
        self.flag_inside_n=~(self.flag_outside_n)

        self.flag_outside_n_mat=np.reshape(self.flag_outside_n,(self.Nyg,self.Nxg),'F');
        self.flag_outside_n_mat=self.flag_outside_n_mat.T
        [gx,gy]=np.gradient(np.double(self.flag_outside_n_mat));
        gradmod=abs(gx)+abs(gy);
        self.flag_border_mat=np.logical_and((gradmod>0), self.flag_outside_n_mat);
        self.flag_border_n = self.flag_border_mat.flatten()

        A = self.assemble_laplacian()

        diagonal = A.diagonal()
        N_full = len(diagonal)
        indices_non_id = np.where(diagonal!=1.)[0]
        N_sel = len(indices_non_id)

        Msel = sps.lil_matrix((N_full, N_sel))
        for ii, ind in enumerate(indices_non_id):
                Msel[ind, ii] =1.
        Msel = Msel.tocsc()

        Asel = Msel.T*A*Msel
        Asel=Asel.tocsc()
        if sparse_solver == 'scipy_slu':
            print "Using scipy superlu solver..."
            self.luobj = spl.splu(Asel.tocsc())
        elif sparse_solver == 'PyKLU':
            print "Using klu solver..."
            try:
                import PyKLU.klu as klu
                self.luobj = klu.Klu(Asel.tocsc())
            except StandardError, e:
                print "Got exception: ", e
                print "Falling back on scipy superlu solver:"
                self.luobj = ssl.splu(Asel.tocsc())
        else:
            raise ValueError('Solver not recognized!!!!\nsparse_solver must be "scipy_klu" or "PyKLU"\n')
        self.Msel = Msel.tocsc()
        self.Msel_T = (Msel.T).tocsc()
        print 'Done PIC init.'

    def assemble_laplacian(self):
        ''' assembles the laplacian and returns the resulting matrix in
        csr format
        '''
        Nxg = self.Nxg
        Nyg = self.Nyg
        Dh = self.Dh
        flag_inside_n = self.flag_inside_n
        A=sps.lil_matrix((Nxg*Nyg,Nxg*Nyg));
        for u in range(0,Nxg*Nyg):
            if np.mod(u, Nxg*Nyg/20)==0:
                print ('Mat. assembly %.0f'%(float(u)/ float(Nxg*Nyg)*100)+"""%""")
            if flag_inside_n[u]:
                A[u,u] = -(4./(Dh*Dh))
                A[u,u-1]=1./(Dh*Dh);     #phi(i-1,j)nx
                A[u,u+1]=1./(Dh*Dh);     #phi(i+1,j)
                A[u,u-Nyg]=1./(Dh*Dh);    #phi(i,j-1)
                A[u,u+Nyg]=1./(Dh*Dh);    #phi(i,j+1)
            else:
                # external nodes
                A[u,u]=1.
        A = A.tocsr()
        return A


    def poisson_solve(self, rho):
        print 'poisson_solve of FiniteDifferences_Staircase_SquareGrid'
        b=-rho.flatten()/epsilon_0;
        b[~(self.flag_inside_n)]=0.; #boundary condition
        b_sel = self.Msel_T*b
        phi_sel = self.luobj.solve(b_sel)
        phi = self.Msel*phi_sel
        return phi





