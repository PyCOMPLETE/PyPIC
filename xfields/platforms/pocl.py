import numpy as np

import pyopencl as cl
import pyopencl.array as cla

from .default_kernels import pocl_default_kernels

class MinimalDotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

class XfPoclPlatform(object):


    def __init__(self, pocl_context=None, command_queue=None, default_kernels=True):

        if pocl_context is None:
            pocl_context = cl.create_some_context()

        if command_queue is None:
            command_queue = cl.CommandQueue(pocl_context)

        assert command_queue.context == pocl_context

        self.pocl_context = pocl_context
        self.command_queue = command_queue
        self.kernels = MinimalDotDict()
        self.FFTClass = XfPoclFFT

        if default_kernels:
            self.add_kernels(src_files=pocl_default_kernels['src_files'],
                    kernel_descriptions=pocl_default_kernels['kernel_descriptions'])


    def nparray_to_platform_mem(self, arr):
        dev_arr = cla.to_device(self.command_queue, arr)
        return dev_arr

    def nparray_from_platform_mem(self, dev_arr):
        return dev_arr.get()

    def add_kernels(self, src_code='', src_files=[], kernel_descriptions={}):

        src_content = src_code
        for ff in src_files:
            with open(ff, 'r') as fid:
                src_content += ('\n\n' + fid.read())

        prg = cl.Program(self.pocl_context, src_content).build()

        ker_names = kernel_descriptions.keys()
        for nn in ker_names:
            kk = getattr(prg, nn)
            aa = kernel_descriptions[nn]['args']
            nt_from = kernel_descriptions[nn]['num_threads_from_arg']
            aa_types, aa_names = zip(*aa)
            self.kernels[nn] = XfPoclKernel(pocl_kernel=kk,
                arg_names=aa_names, arg_types=aa_types,
                num_threads_from_arg=nt_from,
                command_queue=self.command_queue)

class XfPoclKernel(object):

    def __init__(self, pocl_kernel, arg_names, arg_types,
                 num_threads_from_arg, command_queue,
                 wait_on_call=True):

        assert (len(arg_names) == len(arg_types) == pocl_kernel.num_args)
        assert num_threads_from_arg in arg_names

        self.pocl_kernel = pocl_kernel
        self.arg_names = arg_names
        self.arg_types = arg_types
        self.num_threads_from_arg = num_threads_from_arg
        self.command_queue = command_queue
        self.wait_on_call = wait_on_call

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
                assert isinstance(vv, cla.Array)
                assert vv.context == self.pocl_kernel.context
                arg_list.append(vv.base_data[vv.offset:])
            else:
                raise ValueError(f'Type {tt} not recognized')

        event = self.pocl_kernel(self.command_queue,
                (kwargs[self.num_threads_from_arg],),
                None, *arg_list)

        if self.wait_on_call:
            event.wait()

        return event

class XfPoclFFT(object):
    def __init__(self, platform, data, axes):

        self.platform = platform
        self.axes = axes

        import gpyfft
        self._fftobj = gpyfft.fft.FFT(platform.pocl_context,
                platform.command_queue, data, axes)

    def transform(self, data):
        """The transform is done inplace"""

        event, = fftobj.enqueue_arrays(data)
        event.wait()

    def itransform(self, data):
        """The transform is done inplace"""

        event, = fftobj.enqueue_arrays(data, forward=False)
        event.wait()
