import numpy as np
import scipy.constants as cst
cimport numpy as np
from libcpp cimport bool


cdef extern from "FPPSWrapper.h":
    cdef cppclass FPPSWrapper:
        void useSourceAsProbe()
        void scatter(double* x,double* y,double* nel_part,int n)
        void gather(double* x,double* y,double* Ex, double* Ey,int n)
        void solve()

cdef extern from "FPPSWrapper.h":
    cdef cppclass FPPSOpenBoundary(FPPSWrapper):
        FPPSOpenBoundary(int nTheta, int nR, double a) except + #propagates the exception correctly

cdef extern from "FPPSWrapper.h":
    cdef cppclass FPPSUniform(FPPSWrapper):
        FPPSUniform(int nTheta, int nR, double r) except + #propagates the exception correctly
    
cdef class PyFPPS:
    cdef FPPSWrapper *thisptr      # hold a C++ instance which we're wrapping
    cdef double particleCharge
    def __cinit__(self, int nTheta, int nR, double a,bool useSourceAsProbe=False,solverType = 'Uniform'):
        if solverType == 'Uniform':
            self.thisptr = new FPPSUniform(nTheta, nR, a)
        elif solverType == 'OpenBoundary':
            self.thisptr = new FPPSOpenBoundary(nTheta, nR, a)
        else:
            raise Exception(solverType+' is not a solver type')
        if useSourceAsProbe:
            self.thisptr.useSourceAsProbe()

    cpdef scatter(self, np.ndarray x, np.ndarray y, np.ndarray nel_part,double charge=cst.e):
        self.particleCharge = -charge # PyPIC convention ?

        cdef double* x_data = <double*>x.data
        cdef double* y_data = <double*>y.data
        cdef double* nel_part_data = <double*>nel_part.data

        self.thisptr.scatter(x_data,y_data,nel_part_data,len(x))

    cpdef gather(self, np.ndarray x, np.ndarray y):
        cdef double* x_data = <double*>x.data
        cdef double* y_data = <double*>y.data
        cdef np.ndarray Ex = 0.*x; 
        cdef np.ndarray Ey = 0.*x;
        
        cdef double* Ex_data = <double*>Ex.data
        cdef double* Ey_data = <double*>Ey.data

        self.thisptr.gather(x_data, y_data, Ex_data, Ey_data,len(x))
        return Ex*self.particleCharge, Ey*self.particleCharge

    cpdef solve(self):
        self.thisptr.solve()

