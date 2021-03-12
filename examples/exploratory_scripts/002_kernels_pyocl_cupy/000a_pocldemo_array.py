import pyopencl as cl
import pyopencl.array as cl_array
import numpy
import numpy.linalg as la

a = numpy.random.rand(50000).astype(numpy.float32)
b = numpy.random.rand(50000).astype(numpy.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

a_dev = cl_array.to_device(queue, a)
b_dev = cl_array.to_device(queue, b)
dest_dev = cl_array.empty_like(a_dev)

# Here he makes the sum of the two arrays 
# with an explicit kernel                 
prg = cl.Program(ctx, """
    __kernel void sum(__global const float *a,
    __global const float *b, __global float *c)
    {
      int gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }
    """).build()

knl = prg.sum  # Use this Kernel object for repeated calls
knl(queue, a.shape, None, a_dev.data, b_dev.data, dest_dev.data)
# The second argument gives the size od the computing grid (number of threads)
# See here: https://documen.tician.de/pyopencl/runtime_program.html#pyopencl.Kernel.__call__

# Here he compares the result with the operation done with the numpy syntax
print(la.norm((dest_dev - (a_dev+b_dev)).get()))
