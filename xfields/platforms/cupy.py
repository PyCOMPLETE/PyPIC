import numpy as np

import cupy

from .default_kernels import cupy_default_kernels

class MinimalDotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

class XfCupyPlatform(object):

    def __init__(self, default_kernels=True, default_block_size=256):

        'The device can be selected globally using cupy.cuda.Device'

        self.default_block_size = default_block_size

        if default_kernels:
            self.add_kernels(src_files=cupy_default_kernels['src_files'],
                    kernel_descriptions=cupy_default_kernels['kernel_descriptions'])


    def nparray_to_platform_mem(self, arr):
        dev_arr = cupy.array(arr)
        return dev_arr

    def nparray_from_platform_mem(self, dev_arr):
        return dev_arr.get()

    def plan_FFT(self, data, axes, ):
        return XfPoclFFT(self, data, axes)

    def add_kernels(self, src_code='', src_files=[], kernel_descriptions={}):

        src_content = 'extern "C"{'
        for ff in src_files:
            with open(ff, 'r') as fid:
                src_content += ('\n\n' + fid.read())
        src_content += "}"

        module = cupy.RawModule(code=src_content)

        ker_names = kernel_descriptions.keys()
        for nn in ker_names:
            kk = module.get_function(nn)
            aa = kernel_descriptions[nn]['args']
            nt_from = kernel_descriptions[nn]['num_threads_from_arg']
            aa_types, aa_names = zip(*aa)
            self.kernels[nn] = XfCupyKernel(cupy_kernel=kk,
                arg_names=aa_names, arg_types=aa_types,
                num_threads_from_arg=nt_from,
                block_size=self.default_block_size)

class XfCupyKernel(object):

    def __init__(self, cupy_kernel, arg_names, arg_types,
                 num_threads_from_arg, block_size):

        assert (len(arg_names) == len(arg_types) == cupy_kernel.num_args)
        assert num_threads_from_arg in arg_names

        self.cupy_kernel = cupy_kernel
        self.arg_names = arg_names
        self.arg_types = arg_types
        self.num_threads_from_arg = num_threads_from_arg
        self.block_size = block_size

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

        n_threads = kwargs[self.num_threads_from_arg]
        grid_size = int(np.ceil(n_threads/self.block_size))
        self.cupy_kernel(((grid_size, ), (block_size, ), arg_list)


class XfCupyFFT(object):
    def __init__(self, platform, data, axes, wait_on_call=True):

        self.platform = platform
        self.axes = axes
        self.wait_on_call = wait_on_call

        assert len(data.shape) > max(axes)

        # Check internal dimensions are powers of two
        for ii in axes[:-1]:
            nn = data.shape[ii]
            frac_part, _ = np.modf(np.log(nn)/np.log(2))
            assert np.isclose(frac_part, 0) , ('PyOpenCL FFT requires'
                    ' all dimensions apart from the last to be powers of two!')

        import gpyfft
        self._fftobj = gpyfft.fft.FFT(platform.pocl_context,
                platform.command_queue, data, axes=axes)

    def transform(self, data):
        """The transform is done inplace"""

        event, = self._fftobj.enqueue_arrays(data)
        if self.wait_on_call:
            event.wait()
        return event

    def itransform(self, data):
        """The transform is done inplace"""

        event, = self._fftobj.enqueue_arrays(data, forward=False)
        if self.wait_on_call:
            event.wait()
        return event
