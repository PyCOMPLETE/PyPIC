import ctypes

import numpy as np

from .default_kernels import cpu_default_kernels

class MinimalDotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

class XfCpuPlatform(object):

    def __init__(self, default_kernels=True):

        self.kernels = MinimalDotDict()

        if default_kernels:
            self.add_kernels(lib_file=cpu_default_kernels['lib_file'],
                    kernel_descriptions=cpu_default_kernels['kernel_descriptions'])


    def nparray_to_platform_mem(self, arr):
        return arr

    def nparray_from_platform_mem(self, dev_arr):
        return dev_arr

    def add_kernels(self, lib_file, kernel_descriptions={}):

        lib = ctypes.CDLL(lib_file)

        ker_names = kernel_descriptions.keys()
        for nn in ker_names:
            kk = getattr(lib, nn)
            aa = kernel_descriptions[nn]['args']
            aa_types, aa_names = zip(*aa)
            self.kernels[nn] = XfCpuKernel(ctypes_kernel=kk,
                arg_names=aa_names, arg_types=aa_types)

class XfCpuKernel(object):

    def __init__(self, ctypes_kernel, arg_names, arg_types):

        assert (len(arg_names) == len(arg_types))

        self.ctypes_kernel = ctypes_kernel
        self.arg_names = arg_names
        self.arg_types = arg_types

        ct_argtypes = []
        for tt in arg_types:
            if tt[0] == 'scalar':
                ct_argtypes.append(np.ctypeslib.as_ctypes_type(tt[1]))
            elif tt[0] == 'array':
                ct_argtypes.append(np.ctypeslib.ndpointer(dtype=tt[1]))
            else:
                raise ValueError(f'Type {tt} not recognized')
            self.ctypes_kernel.argtypes = ct_argtypes

    @property
    def num_args(self):
        return len(self.arg_names)

    def __call__(self, **kwargs):
        assert len(kwargs.keys()) == self.num_args
        arg_list = []
        for nn, tt in zip(self.arg_names, self.arg_types):
            vv = kwargs[nn]
            if tt[0] == 'scalar':
                assert np.isscalar(vv)
                arg_list.append(tt[1](vv))
            elif tt[0] == 'array':
                arg_list.append(vv)
            else:
                raise ValueError(f'Type {tt} not recognized')

        event = self.ctypes_kernel(*arg_list)
