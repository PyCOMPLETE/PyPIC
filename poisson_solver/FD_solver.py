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
                                yg=None, ext_boundary=False):
    '''Function which returns (Dh, xg, Nxg, bias_x, yg, Nyg, bias_y)
    Guarantees backwards compatibility
    '''
    #TODO put this into the PyPIC class, the Solver should use the grid it
    # gets and not change it!
    if ext_boundary:
        x_aper += 5.*Dh
        y_aper += 4.*Dh
    else:
        x_aper += 1e-10*Dh
        y_aper += 1e-10*Dh
    if xg!=None and yg!=None:
        assert(x_aper==None and y_aper==None and Dh==None)
        Nxg=len(xg);
        bias_x=min(xg);
        Nyg=len(yg);
        bias_y=min(yg);
        Dh = xg[1]-xg[0]
    else:
        assert(xg==None and yg==None)
        xg=np.arange(0, x_aper,Dh,float)
        xgr=xg[1:]
        xgr=xgr[::-1]#reverse array
        xg=np.concatenate((-xgr,xg),0)
        Nxg=len(xg);
        bias_x=min(xg);
        yg=np.arange(0,y_aper,Dh,float)
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
    def __init__(self, chamb, Dh, sparse_solver='scipy_slu', ext_boundary=False):
        # mimics the super() call in the old version
        params = compute_new_mesh_properties(
                chamb.x_aper, chamb.y_aper, Dh, ext_boundary=ext_boundary)
        self.Dh, self.xg, self.Nxg, self.bias_x, self.yg, self.Nyg, self.bias_y = params
        self.chamb = chamb

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
        for u in xrange(0,Nxg*Nyg):
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


    def poisson_solve(self, mesh_charges):
        rho = mesh_charges.reshape(self.Nyg, self.Nxg).T / (self.Dh*self.Dh)
        b=-rho.flatten()/epsilon_0;
        b[~(self.flag_inside_n)]=0.; #boundary condition
        #TODO debug only
        self.b = b
        b_sel = self.Msel_T*b
        phi_sel = self.luobj.solve(b_sel)
        phi = self.Msel*phi_sel
        return phi.reshape(self.Nxg, self.Nyg).T.flatten()


class FiniteDifferences_ShortleyWeller_SquareGrid(FiniteDifferences_Staircase_SquareGrid):
    '''
    TODO
    '''
    #TODO: check how we want to use the Dx/Dy matrices (efx = Dx*phi...)
    def __init__(self, chamb, Dh, sparse_solver='scipy_slu'):
        #do all the basic stuff, including creating the A matrix by using the
        #assemble_laplacian function from this child class. the only thing
        # to be done in this constructor is the construction of the Dx,Dy mats
        # luckily this also happens in the assemble_laplacian of this class
        # which sets the matrices self.Dx, self.Dy
        super(FiniteDifferences_ShortleyWeller_SquareGrid, self).__init__(
                chamb, Dh, sparse_solver)


    def assemble_laplacian(self):
        '''override!
        Returns the matrix A arising from the discretisation
        sets self.Dx, self.Dy (matrices due to shortley weller)
        '''
        na = lambda x: np.array([x])
        Nxg = self.Nxg
        Nyg = self.Nyg
        chamb = self.chamb
        Dx=sps.lil_matrix((Nxg*Nyg,Nxg*Nyg));
        Dy=sps.lil_matrix((Nxg*Nyg,Nxg*Nyg));
        A=sps.lil_matrix((Nxg*Nyg,Nxg*Nyg));
        flag_inside_n = self.flag_inside_n
        xn = self.xn
        yn = self.yn
        Dh = self.Dh
        for u in xrange(0,Nxg*Nyg):
            if np.mod(u, Nxg*Nyg/20)==0:
                print ('Mat. assembly %.0f'%(float(u)/ float(Nxg*Nyg)*100)+"""%""")
            if flag_inside_n[u]:
                #Compute Shortley-Weller coefficients
                if flag_inside_n[u-1]: #phi(i-1,j)
                    hw = Dh
                else:
                    x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u-1]), na(yn[u-1]), na(0.), resc_fac=.995, flag_robust=False)
                    hw = np.abs(y_int[0]-yn[u])
                if flag_inside_n[u+1]: #phi(i+1,j)
                    he = Dh
                else:
                    x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u+1]), na(yn[u+1]), na(0.), resc_fac=.995, flag_robust=False)
                    he = np.abs(y_int[0]-yn[u])
                if flag_inside_n[u-Nyg]: #phi(i,j-1)
                    hs = Dh
                else:
                    x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u-Nyg]), na(yn[u-Nyg]), na(0.), resc_fac=.995, flag_robust=False)
                    hs = np.abs(x_int[0]-xn[u])
                    #~ print hs
                if flag_inside_n[u+Nyg]: #phi(i,j+1)
                    hn = Dh
                else:
                    x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u+Nyg]), na(yn[u+Nyg]), na(0.), resc_fac=.995, flag_robust=False)
                    hn = np.abs(x_int[0]-xn[u])
                    #~ print hn

                # Build A matrix
                if hn<Dh/100. or hs<Dh/100. or hw<Dh/100. or he<Dh/100.: # nodes very close to the bounday
                    A[u,u] =1.
                    #list_internal_force_zero.append(u)
                #print u, xn[u], yn[u]
                else:
                    A[u,u] = -(2./(he*hw)+2/(hs*hn))
                    A[u,u-1]=2./(hw*(hw+he));     #phi(i-1,j)nx
                    A[u,u+1]=2./(he*(hw+he));     #phi(i+1,j)
                    A[u,u-Nyg]=2./(hs*(hs+hn));    #phi(i,j-1)
                    A[u,u+Nyg]=2./(hn*(hs+hn));    #phi(i,j+1)

                # Build Dx matrix
                if hn<Dh/100.:
                    if hs>=Dh/100.:
                        Dx[u,u] = -1./hs
                        Dx[u,u-Nyg]=1./hs
                elif hs<Dh/100.:
                    if hn>=Dh/100.:
                        Dx[u,u] = 1./hn
                        Dx[u,u+Nyg]=-1./hn
                else:
                    Dx[u,u] = (1./(2*hn)-1./(2*hs))
                    Dx[u,u-Nyg]=1./(2*hs)
                    Dx[u,u+Nyg]=-1./(2*hn)


                # Build Dy matrix
                if he<Dh/100.:
                    if hw>=Dh/100.:
                        Dy[u,u] = -1./hw
                        Dy[u,u-1]=1./hw
                elif hw<Dh/100.:
                    if he>=Dh/100.:
                        Dy[u,u] = 1./he
                        Dy[u,u+1]=-1./(he)
                else:
                    Dy[u,u] = (1./(2*he)-1./(2*hw))
                    Dy[u,u-1]=1./(2*hw)
                    Dy[u,u+1]=-1./(2*he)
            else:
                # external nodes
                A[u,u]=1.
        self.Dx = Dx.tocsc()
        self.Dy = Dy.tocsc()
        return A.tocsc()

    def gradient(self, dummy):
        ''' Function which returns a function to compute the gradient specific
        for this Shortley Weller approximation
        '''
        def _gradient(phi):
            efx = self.Dx*phi
            efy = self.Dy*phi
            efx, efy = efy, efx  # something is wrong...
            return [efx, efy]
        return _gradient

class FiniteDifferences_ShortleyWeller_SquareGrid_extrapolation(FiniteDifferences_Staircase_SquareGrid):
    '''
    '''
    def __init__(self, chamb, Dh, sparse_solver='scipy_slu'):
        super(FiniteDifferences_ShortleyWeller_SquareGrid_extrapolation, self).__init__(
                chamb, Dh, sparse_solver, ext_boundary=True)

    def handle_border(self, u, flag_inside_n, Nxg, Nyg, xn, yn, chamb, Dh, Dx, Dy):
        #print u

        na = lambda x: np.array([x])
        jjj = np.floor(u/Nyg)

        if flag_inside_n[u+Nyg]:
            if not flag_inside_n[u]:
                x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal( na(xn[u+Nyg]), na(yn[u+Nyg]), na(0.),
                        na(xn[u]), na(yn[u]), na(0.), resc_fac=.995, flag_robust=False)
                hs = np.abs(x_int[0]-xn[u+Nyg])
            else: #this is the case for internal nodes with zero potential (very close to the boundary)
                hs = Dh

            hn = Dh

            if hs<Dh/100.:
                Dx[u,u+Nyg] = (1./(hn))
                Dx[u,u+Nyg+Nyg]=-1./(hn)

                nnn=1
                while u-nnn*Nyg>=0:
                    Dx[u-nnn*Nyg,u+Nyg] = (1./(hn))
                    Dx[u-nnn*Nyg,u+Nyg+Nyg]=-1./(hn)
                    nnn+=1

            else:
                Dx[u,u+Nyg] = (1./(2*hn)-1./(2*hs))
                Dx[u,u-Nyg+Nyg] = 1./(2*hs)
                Dx[u,u+Nyg+Nyg] = -1./(2*hn)

                nnn=1
                while u-nnn*Nyg>=0:
                    Dx[u-nnn*Nyg,u+Nyg] = Dx[u,u+Nyg]
                    Dx[u-nnn*Nyg,u-Nyg+Nyg] = Dx[u,u-Nyg+Nyg]
                    Dx[u-nnn*Nyg,u+Nyg+Nyg] = Dx[u,u+Nyg+Nyg]
                    nnn+=1


        elif flag_inside_n[u-Nyg]:
            if not flag_inside_n[u]:
                x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal( na(xn[u-Nyg]), na(yn[u-Nyg]), na(0.),
                        na(xn[u]), na(yn[u]), na(0.), resc_fac=.995, flag_robust=False)
                hn = np.abs(x_int[0]-xn[u-Nyg])
            else:#this is the case for internal nodes with zero potential (very close to the boundary)
                hn = Dh

            hs = Dh

            if hn<Dh/100.:
                Dx[u,u-Nyg] = -1./(hs)
                Dx[u,u-Nyg-Nyg]=1./(hs)

                nnn=1
                while u+nnn*Nyg<Nxg*Nyg:
                    Dx[u+nnn*Nyg,u-Nyg] = -1./(hs)
                    Dx[u+nnn*Nyg,u-Nyg-Nyg]=1./(hs)
                    nnn+=1

            else:
                Dx[u,u-Nyg] = (1./(2*hn)-1./(2*hs))
                Dx[u,u-Nyg-Nyg]=1./(2*hs)
                Dx[u,u+Nyg-Nyg]=-1./(2*hn)

                nnn=1
                while u+nnn*Nyg<Nxg*Nyg:
                    Dx[u+nnn*Nyg,u-Nyg] = Dx[u,u-Nyg]
                    Dx[u+nnn*Nyg,u-Nyg-Nyg] = Dx[u,u-Nyg-Nyg]
                    Dx[u+nnn*Nyg,u+Nyg-Nyg] = Dx[u,u+Nyg-Nyg]
                    nnn+=1

        if flag_inside_n[u+1]:
            if not flag_inside_n[u]:
                x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal( na(xn[u+1]), na(yn[u+1]), na(0.),
                        na(xn[u]), na(yn[u]),na(0.), resc_fac=.995, flag_robust=False)
                hw = np.abs(y_int[0]-yn[u+1])
            else:#this is the case for internal nodes with zero potential (very close to the boundary)
                hw = Dh

            he = Dh

            if hw<Dh/100.:
                Dy[u,u+1] = (1./(he))
                Dy[u,u+1+1]=-1./(he)

                nnn=1
                while u-nnn>=(jjj)*Nyg:
                    Dy[u-nnn*1,u+1] = (1./(he))
                    Dy[u-nnn*1,u+1+1]=-1./(he)
                    nnn+=1
            else:
                Dy[u,u+1] = (1./(2*he)-1./(2*hw))
                Dy[u,u-1+1] = 1./(2*hw)
                Dy[u,u+1+1] = -1./(2*he)

                nnn=1
                while u-nnn>=(jjj)*Nyg:
                    #print nnn
                    Dy[u-nnn,u+1] = Dy[u,u+1]
                    Dy[u-nnn,u-1+1] = Dy[u,u-1+1]
                    Dy[u-nnn,u+1+1] = Dy[u,u+1+1]
                    nnn += 1

        elif flag_inside_n[u-1]:
            if not flag_inside_n[u]:
                x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal( na(xn[u-1]), na(yn[u-1]), na(0.),
                        na(xn[u]), na(yn[u]),  na(0.), resc_fac=.995, flag_robust=False)
                he = np.abs(y_int[0]-yn[u-1])
            else:#this is the case for internal nodes with zero potential (very close to the boundary)
                he=Dh

            hw = Dh

            if he<Dh/100.:
                Dy[u,u-1] = -1./(hw)
                Dy[u,u-1-1]=1./(hw)

                nnn=1
                while u+nnn<(jjj+1)*Nyg:
                    Dy[u+nnn,u-1] = -1./(hw)
                    Dy[u+nnn,u-1-1]=1./(hw)
                    nnn+=1

            else:
                Dy[u,u-1] = (1./(2*he)-1./(2*hw))
                Dy[u,u-1-1]=1./(2*hw)
                Dy[u,u+1-1]=-1./(2*he)

                nnn=1
                while u+nnn<(jjj+1)*Nyg:
                    Dy[u+nnn,u-1] = Dy[u,u-1]
                    Dy[u+nnn,u-1-1] = Dy[u,u-1-1]
                    Dy[u+nnn,u+1-1] = Dy[u,u+1-1]
                    nnn+=1
        return Dx, Dy


    def assemble_laplacian(self):
        ''' override '''
        na = lambda x: np.array([x])
        Nxg = self.Nxg
        Nyg = self.Nyg
        chamb = self.chamb
        self.Dx=sps.lil_matrix((Nxg*Nyg,Nxg*Nyg));
        self.Dy=sps.lil_matrix((Nxg*Nyg,Nxg*Nyg));
        A=sps.lil_matrix((Nxg*Nyg,Nxg*Nyg));
        flag_inside_n = self.flag_inside_n
        xn = self.xn
        yn = self.yn
        Dh = self.Dh
        Dx = self.Dx
        Dy = self.Dy
        list_internal_force_zero = []

        # Build A Dx Dy matrices
        for u in xrange(0,Nxg*Nyg):
            if np.mod(u, Nxg*Nyg/20)==0:
                    print ('Mat. assembly %.0f'%(float(u)/ float(Nxg*Nyg)*100)+"""%""")
            if flag_inside_n[u]:

                #Compute Shortley-Weller coefficients
                if flag_inside_n[u-1]: #phi(i-1,j)
                    hw = Dh
                else:
                    x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u-1]), na(yn[u-1]), na(0.), resc_fac=.995, flag_robust=False)
                    hw = np.abs(y_int[0]-yn[u])

                if flag_inside_n[u+1]: #phi(i+1,j)
                    he = Dh
                else:
                    x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u+1]), na(yn[u+1]), na(0.), resc_fac=.995, flag_robust=False)
                    he = np.abs(y_int[0]-yn[u])

                if flag_inside_n[u-Nyg]: #phi(i,j-1)
                    hs = Dh
                else:
                    x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u-Nyg]), na(yn[u-Nyg]), na(0.), resc_fac=.995, flag_robust=False)
                    hs = np.abs(x_int[0]-xn[u])
                    #~ print hs

                if flag_inside_n[u+Nyg]: #phi(i,j+1)
                    hn = Dh
                else:
                    x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u+Nyg]), na(yn[u+Nyg]), na(0.), resc_fac=.995, flag_robust=False)
                    hn = np.abs(x_int[0]-xn[u])
                    #~ print hn

                # Build A matrix
                if hn<Dh/100. or hs<Dh/100. or hw<Dh/100. or he<Dh/100.: # nodes very close to the bounday
                    A[u,u] =1.
                    list_internal_force_zero.append(u)
                    #print u, xn[u], yn[u]
                else:
                    A[u,u] = -(2./(he*hw)+2/(hs*hn))
                    A[u,u-1]=2./(hw*(hw+he));     #phi(i-1,j)nx
                    A[u,u+1]=2./(he*(hw+he));     #phi(i+1,j)
                    A[u,u-Nyg]=2./(hs*(hs+hn));    #phi(i,j-1)
                    A[u,u+Nyg]=2./(hn*(hs+hn));    #phi(i,j+1)

                # Build Dx matrix
                if hn<Dh/100.:
                    if hs>=Dh/100.:
                        Dx[u,u] = -1./hs
                        Dx[u,u-Nyg]=1./hs
                elif hs<Dh/100.:
                    if hn>=Dh/100.:
                        Dx[u,u] = 1./hn
                        Dx[u,u+Nyg]=-1./hn
                else:
                    Dx[u,u] = (1./(2*hn)-1./(2*hs))
                    Dx[u,u-Nyg]=1./(2*hs)
                    Dx[u,u+Nyg]=-1./(2*hn)


                # Build Dy matrix
                if he<Dh/100.:
                    if hw>=Dh/100.:
                        Dy[u,u] = -1./hw
                        Dy[u,u-1]=1./hw
                elif hw<Dh/100.:
                    if he>=Dh/100.:
                        Dy[u,u] = 1./he
                        Dy[u,u+1]=-1./(he)
                    else:
                        Dy[u,u] = (1./(2*he)-1./(2*hw))
                        Dy[u,u-1]=1./(2*hw)
                        Dy[u,u+1]=-1./(2*he)

            else:
                # external nodes
                A[u,u]=1.
                if self.flag_border_n[u]:
                    self.handle_border(u, self.flag_inside_n, self.Nxg,
                                       self.Nyg, self.xn, self.yn, self.chamb,
                                       self.Dh, self.Dx, self.Dy)

        for u in list_internal_force_zero:
            self.handle_border(u, self.flag_inside_n, self.Nxg,
                               self.Nyg, self.xn, self.yn, self.chamb, self.Dh,
                               self.Dx, self.Dy)

        #~ A = A.tocsc()
        self.Dx = self.Dx.tocsc()
        self.Dy = self.Dy.tocsc()

        flag_force_zero = self.flag_outside_n.copy()
        for ind in  list_internal_force_zero:
            flag_force_zero[ind] = True

        flag_force_zero_mat=np.reshape(flag_force_zero,(self.Nyg,self.Nxg),'F');
        flag_force_zero_mat=flag_force_zero_mat.T
        [gxc,gyc]=np.gradient(np.double(flag_force_zero_mat));
        gradmodc=abs(gxc)+abs(gyc);
        flag_border_mat_c=np.logical_and((gradmodc>0), flag_force_zero_mat);

        sumcurr = np.sum(flag_border_mat_c, axis=0)
        self.jj_max_border = np.max((np.where(sumcurr>0))[0])
        self.jj_min_border = np.min((np.where(sumcurr>0))[0])

        sumcurr = np.sum(flag_border_mat_c, axis=1)
        self.ii_max_border = np.max((np.where(sumcurr>0))[0])
        self.ii_min_border = np.min((np.where(sumcurr>0))[0])

        return A.tocsc()

    def gradient(self, dummy):
        ''' Function which returns a function to compute the gradient specific
        to this Shortley Weller approximation
        '''
        def _gradient(phi):
            efx = self.Dx*phi
            efy = self.Dy*phi
            efx=np.reshape(efx, (self.Nxg, self.Nyg))
            efy=np.reshape(efy, (self.Nxg, self.Nyg))
            for jj in xrange(self.jj_max_border, self.Nyg):
                efx[:, jj]=efx[:, self.jj_max_border-1]
            for jj in xrange(0, self.jj_min_border+1):
                efx[:, jj]=efx[:, self.jj_min_border+1]
            for ii in xrange(self.ii_max_border, self.Nxg):
                efy[ii, :]=efy[self.ii_max_border-1, :]
            for ii in xrange(0, self.ii_min_border+1):
                efy[ii,:]=efy[self.ii_min_border+1,:]
            efx = efx.flatten()
            efy = efy.flatten()
            efx, efy = efy, efx
            return [efx, efy]
        return _gradient


