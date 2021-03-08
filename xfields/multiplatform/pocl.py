import numpy as np
import pyopencl.array as cla

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
            if np.issctype(tt):
                assert np.isscalar(vv)
                arg_list.append(tt(vv))
            else:
                assert isinstance(vv, cla.Array)
                assert vv.context == self.pocl_kernel.context
                arg_list.append(vv.base_data[vv.offset:])

        event = self.pocl_kernel(self.command_queue,
                (kwargs[self.num_threads_from_arg],),
                None, *arg_list)

        if self.wait_on_call:
            event.wait()

        return event
