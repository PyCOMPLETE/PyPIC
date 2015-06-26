"""
cuSOLVER RF Wrapper for python using ctypes
@author Stefan Hegglin
"""

import ctypes


CUSOLVER_STATUS = {
    1: "CUSOLVER_STATUS_NOT_INITIALIZED",
    2: "CUSOLVER_STATUS_ALLOC_FAILED",
    3: "CUSOLVER_STATUS_INVALID_VALUE",
    4: "CUSOLVER_STATUS_ARCH_MISMATCH",
    5: "CUSOLVER_STATUS_MAPPING_ERROR",
    6: "CUSOLVER_STATUS_EXECUTION_FAILED",
    7: "CUSOLVER_STATUS_INTERNAL_ERROR",
    8: "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED",
    9: "CUSOLVER_STATUS_NOT_SUPPORTED",
    10: "CUSOLVER_STATUS_ZERO_PIVOT",
    11: "CUSOLVER_STATUS_INVALID_LICENSE"
}

CUSOLVERRF_MATRIXFORMAT_T = {
    0: "CUSOLVERRF_MATRIX_FORMAT_CSR",
    1: "CUSOLVERRF_MATRIX_FORMAT_CSC"
}

CUSOLVERRF_UNITDIAGONAL_T = {
    0: "CUSOLVERRF_UNIT_DIAGONAL_STORED_L",
    1: "CUSOLVERRF_UNIT_DIAGONAL_STORED_U",
    2: "CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L",
    3: "CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U"
}

CUSOLVERRF_FACTORIZATION_ALG_T = {
        0: "CUSOLVERRF_FACTORIZATION_ALG0",
        1: "CUSOLVERRF_FACTORIZATION_ALG1",
        2: "CUSOLVERRF_FACTORIZATION_ALG2"
    }

CUSOLVERRF_TRIANGULAR_SOLVE_T = {
        0: "CUSOLVERRF_TRIANGULAR_SOLVE_ALG0",
        1: "CUSOLVERRF_TRIANGULAR_SOLVE_ALG1",
        2: "CUSOLVERRF_TRIANGULAR_SOLVE_ALG2",
        3: "CUSOLVERRF_TRIANGULAR_SOLVE_ALG3"
    }

CUSOLVERRF_RESET_VALUES_FAST_MODE_T = {
        0: "CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF",
        1: "CUSOLVERRF_RESET_VALUES_FAST_MODE_ON"
}


def cusolver_check_status(status):
    """
    Raise a RuntimeException if status != 0
    """
    if status != 0:
        raise RuntimeError("CUDA returned with status " + str(status) + ": "
                           + CUSOLVER_STATUS[status])

_libcusolver = ctypes.cdll.LoadLibrary('libcusolver.so')



_libcusolver.cusolverRfCreate.restype = int
_libcusolver.cusolverRfCreate.argtypes = [ctypes.c_void_p]
def cusolverRfCreate():
    """
    Returns a handle to a cusolverRf context
    """
    handle = ctypes.c_void_p()
    cusolver_check_status(_libcusolver.cusolverRfCreate(ctypes.byref(handle)))
    return handle.value


_libcusolver.cusolverRfDestroy.restype = int
_libcusolver.cusolverRfDestroy.argtypes = [ctypes.c_void_p]
def cusolverRfDestroy(handle):
    """
    Destroys the context belonging to the handle argument
    """
    cusolver_check_status(_libcusolver.cusolverRfDestroy(handle))

_libcusolver.cusolverRfSetupHost.restype = int
_libcusolver.cusolverRfSetupHost.argtypes = [ctypes.c_int,     # n
                                          ctypes.c_int,     # nnz
                                          ctypes.c_void_p,  # csrRowA
                                          ctypes.c_void_p,  # csrColIndA
                                          ctypes.c_void_p,  # csrValA
                                          ctypes.c_int,     # nnzL
                                          ctypes.c_void_p,  # csrRowPtrL
                                          ctypes.c_void_p,  # csrColIndL
                                          ctypes.c_void_p,  # csrValL
                                          ctypes.c_int,     # nnzU
                                          ctypes.c_void_p,  # csrRowPtrU
                                          ctypes.c_void_p,  # csrColIndU
                                          ctypes.c_void_p,  # csrValU
                                          ctypes.c_void_p,  # P
                                          ctypes.c_void_p,  # Q
                                          ctypes.c_void_p ] # handle
def cusolverRfSetupHost(n, nnzA, csrRowPtrA, csrColIndA, csrValA,
                    nnzL, csrRowPtrL, csrColIndL, csrValL,
                    nnzU, csrRowPtrU, csrColIndU, csrValU,
                    P, Q, handle):
    """ Wraps the cusolverRFSetup function
    All arrays must be on the CPU already
    """
    cusolver_check_status(_libcusolver.cusolverRfSetupHost(
        n,
        nnzA, int(csrRowPtrA.ctypes.data), int(csrColIndA.ctypes.data), int(csrValA.ctypes.data),
        nnzL, int(csrRowPtrL.ctypes.data), int(csrColIndL.ctypes.data), int(csrValL.ctypes.data),
        nnzU, int(csrRowPtrU.ctypes.data), int(csrColIndU.ctypes.data), int(csrValU.ctypes.data),
        int(P.ctypes.data), int(Q.ctypes.data), handle))


_libcusolver.cusolverRfSetupDevice.restype = int
_libcusolver.cusolverRfSetupDevice.argtypes = [ctypes.c_int,     # n
                                          ctypes.c_int,     # nnz
                                          ctypes.c_void_p,  # csrRowA
                                          ctypes.c_void_p,  # csrColIndA
                                          ctypes.c_void_p,  # csrValA
                                          ctypes.c_int,     # nnzL
                                          ctypes.c_void_p,  # csrRowPtrL
                                          ctypes.c_void_p,  # csrColIndL
                                          ctypes.c_void_p,  # csrValL
                                          ctypes.c_int,     # nnzU
                                          ctypes.c_void_p,  # csrRowPtrU
                                          ctypes.c_void_p,  # csrColIndU
                                          ctypes.c_void_p,  # csrValU
                                          ctypes.c_void_p,  # P
                                          ctypes.c_void_p,  # Q
                                          ctypes.c_void_p ] # handle
def cusolverRfSetup(n, nnzA, csrRowPtrA, csrColIndA, csrValA,
                    nnzL, csrRowPtrL, csrColIndL, csrValL,
                    nnzU, csrRowPtrU, csrColIndU, csrValU,
                    P, Q, handle):
    """ Wraps the cusolverRFSetup function
    All arrays must be on the GPU already
    Wrong documentation on cuda website: function is called cusolverRfSetupDevice
    (not cusolverRfSetup)
    """
    cusolver_check_status(_libcusolver.cusolverRfSetupDevice(
        n,
        nnzA, int(csrRowPtrA.gpudata), int(csrColIndA.gpudata), int(csrValA.gpudata),
        nnzL, int(csrRowPtrL.gpudata), int(csrColIndL.gpudata), int(csrValL.gpudata),
        nnzU, int(csrRowPtrU.gpudata), int(csrColIndU.gpudata), int(csrValU.gpudata),
        int(P.gpudata), int(Q.gpudata), handle))


_libcusolver.cusolverRfExtractBundledFactorsHost.restype = int
_libcusolver.cusolverRfExtractBundledFactorsHost.argtypes = [ ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_void_p]
def cusolverRfExtractBundledFactorsHost(handle, nnz_M, csrRowPtrM, csrColIndM,
                                        csrValM):
    """Extracts M=(L-1)+U. Expects prior call to cusolverRfRefactor
    All arrays on Host
    csrRowPtr, ColInd & Val must be pointers!
    """
    cusolver_check_status(_libcusolver.cusolverRfExtractBundledFactorsHost(
        handle, ctypes.byref(nnz_M), ctypes.byref(csrRowPtrM), ctypes.byref(csrColIndM),
        ctypes.byref(csrValM)))


_libcusolver.cusolverRfGetMatrixFormat.restype = ctypes.c_int
_libcusolver.cusolverRfGetMatrixFormat.argtypes = [ ctypes.c_void_p,
                                                    ctypes.c_void_p,
                                                    ctypes.c_void_p]
def cusolverRfGetMatrixFormat(handle, format, diag):
    """gets the matrixformat type and diagonal type
    Currentlyprints the format and the diag type
    arguments format and diag are not used right now
    """
    format = ctypes.c_int(50)
    diag = ctypes.c_int(40)
    cusolver_check_status(_libcusolver.cusolverRfGetMatrixFormat(
        handle, ctypes.byref(format), ctypes.byref(diag)))
    print("format: " + CUSOLVERRF_MATRIXFORMAT_T[int(format.value)])
    print("diag: " + CUSOLVERRF_UNITDIAGONAL_T [int(diag.value)])


_libcusolver.cusolverRfAnalyze.restype = int
_libcusolver.cusolverRfAnalyze.argtypes = [ctypes.c_void_p]
def cusolverRfAnalyze(handle):
    """ Wrapper for cusolverRfAnalyze
    Performs appropriate analyzis of parallelism available in the
    re-factorization of A=L*U. It is assumed cusolverRfSetupHost has been
    called before calling these functions to create the internal
    data structures needed for the analysis.
    """
    cusolver_check_status(_libcusolver.cusolverRfAnalyze(handle))


_libcusolver.cusolverRfRefactor.restype = ctypes.c_int
_libcusolver.cusolverRfRefactor.argtypes = [ctypes.c_void_p]
def cusolverRfRefactor(handle):
    """ Wrapper for cusolverRFRefactor
    Performs LU re-factorization A=L*U exploring the available parallelism
    on the GPU. A prior call to cusolverRfAnalyze() is assumed.
    Call once for each of the linear systems A_i*x_i = f_i
    """
    cusolver_check_status(_libcusolver.cusolverRfRefactor(handle))


_libcusolver.cusolverRfSolve.restype = int
_libcusolver.cusolverRfSolve.argtypes = [ctypes.c_void_p, #handle
                                         ctypes.c_void_p, #P
                                         ctypes.c_void_p, #Q
                                         ctypes.c_int,    #nrhs
                                         ctypes.c_void_p, #Temp
                                         ctypes.c_int,    #ldt
                                         ctypes.c_void_p, #XF
                                         ctypes.c_int]    #ldxf
def cusolverRfSolve(P, Q, nrhs, Temp, ldt, XF, ldxf, handle):
    """Performs the forward and backward solve with the LU factorization
    from a previous call to cusolverRfRefactor().
    P,Q are on the device
    Temp, XF are on the device
    Temp is a buffer of size >= ldt*nrhs >= n*nrhs
    """
    cusolver_check_status(_libcusolver.cusolverRfSolve(
        handle, int(P.gpudata), int(Q.gpudata), nrhs, int(Temp.gpudata),
        ldt, int(XF.gpudata), ldxf)
        )


_libcusolver.cusolverRfSetResetValuesFastMode.restype = int
_libcusolver.cusolverRfSetResetValuesFastMode.argtypes = [ctypes.c_void_p,
                                                          ctypes.c_int]
def cusolverRfSetResetValuesFastMode(handle, mode):
    """Reset value to fast mode
    """
    cusolver_check_status(_libcusolver.cusolverRfSetResetValuesFastMode(
        handle, mode))


_libcusolver.cusolverRfAccessBundledFactorsDevice.restype = int
_libcusolver.cusolverRfAccessBundledFactorsDevice.argtypes = [ctypes.c_void_p,
                                                        ctypes.c_void_p,
                                                        ctypes.c_void_p,
                                                        ctypes.c_void_p,
                                                        ctypes.c_void_p]
def cusolverRfAccessBundledFactors(handle, nnz_M, csrRowPtr, csrColInd, csrVal):
    """Attention: function name is ...Device()!!!"""
    cusolver_check_status(_libcusolver.cusolverRfAccessBundledFactorsDevice(
        handle, ctypes.byref(nnz_M), ctypes.byref(csrRowPtr),
        ctypes.byref(csrColInd), ctypes.byref(csrVal)))


_libcusolver.cusolverRfResetValues.restype = int
_libcusolver.cusolverRfResetValues.argtypes = [ctypes.c_int,
                                               ctypes.c_int,
                                               ctypes.c_void_p,
                                               ctypes.c_void_p,
                                               ctypes.c_void_p,
                                               ctypes.c_void_p,
                                               ctypes.c_void_p,
                                               ctypes.c_void_p]
def cusolverRfResetValues(handle, n, nnzA, RowPtrA, ColIndA, ValA, P, Q):
    """cusolverRfResetValues Wrapper
    All pointers on GPU
    """
    cusolver_check_status(_libcusolver.cusolverRfResetValues(
        n, nnzA, int(RowPtrA.gpudata), int(ColIndA.gpudata), int(ValA.gpudata),
        int(P.gpudata), int(Q.gpudata), handle))


_libcusolver.cusolverRfGetAlgs.restype = ctypes.c_int
_libcusolver.cusolverRfGetAlgs.argtypes = [ ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p]
def cusolverRfGetAlgs(handle, fact_alg, solve_alg):
    """Print the current fact and solve algorithms
    The arguments fact_alg, solve_alg are unused at the moment
    """
    fact_alg = ctypes.c_int()
    solve_alg = ctypes.c_int()
    cusolver_check_status(_libcusolver.cusolverRfGetAlgs(
        handle, ctypes.byref(fact_alg), ctypes.byref(solve_alg)))
    print("fact alg: " + CUSOLVERRF_FACTORIZATION_ALG_T[int(fact_alg.value)])
    print("solve alg: " + CUSOLVERRF_TRIANGULAR_SOLVE_T[int(solve_alg.value)])




############## cusparse

_libcusparse = ctypes.cdll.LoadLibrary('libcusparse.so')

_libcusparse.cusparseCreate.restype = int
_libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]
def cusparseCreate():
    handle = ctypes.c_void_p()
    cusolver_check_status(_libcusparse.cusparseCreate(ctypes.byref(handle)))
    return handle.value

_libcusparse.cusparseDestroy.restype = int
_libcusparse.cusparseDestroy.argtypes = [ctypes.c_void_p]
def cusparseDestroy(handle):
    cusolver_check_status(cusparseDestroy(handle))


_libcusparse.cusparseCreateMatDescr.restype = int
_libcusparse.cusparseCreateMatDescr.argtypes = [ctypes.c_void_p]
def cusparseCreateMatDescr():
    descr = ctypes.c_void_p()
    cusolver_check_status(_libcusparse.cusparseCreateMatDescr(ctypes.byref(descr)))
    return descr


##### cusolverSp

_libcusolver.cusolverSpCreate.restype = int
_libcusolver.cusolverSpCreate.argtypes = [ctypes.c_void_p]
def cusolverSpCreate():
    """
    Returns a handle to a cusolverRf context
    """
    handle = ctypes.c_void_p()
    cusolver_check_status(_libcusolver.cusolverSpCreate(ctypes.byref(handle)))
    return handle.value


_libcusolver.cusolverSpDestroy.restype = int
_libcusolver.cusolverSpDestroy.argtypes = [ctypes.c_void_p]
def cusolverSpDestroy(handle):
    """
    Destroys the context belonging to the handle argument
    """
    cusolver_check_status(_libcusolver.cusolverSpDestroy(handle))



_libcusolver.cusolverSpDcsrlsvchol.restype = int
_libcusolver.cusolverSpDcsrlsvchol.argtypes = [ctypes.c_void_p,
                                               ctypes.c_int,
                                               ctypes.c_int,
                                               ctypes.c_void_p,
                                               ctypes.c_void_p,
                                               ctypes.c_void_p,
                                               ctypes.c_void_p,
                                               ctypes.c_void_p,
                                               ctypes.c_double,
                                               ctypes.c_int,
                                               ctypes.c_void_p,
                                               ctypes.c_void_p ]

def cusolverSpDcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd,
        b, tol, reorder, x, singularity):
    cusolver_check_status(_libcusolver.cusolverSpDcsrlsvchol(
                                handle,
                                m,
                                nnz,
                                descrA,
                                int(csrVal.gpudata),
                                int(csrRowPtr.gpudata),
                                int(csrColInd.gpudata),
                                int(b.gpudata),
                                tol,
                                reorder,
                                int(x.gpudata),
                                ctypes.byref(singularity)))


