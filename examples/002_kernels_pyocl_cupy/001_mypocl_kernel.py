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
src_file = '../../xfields/src_autogenerated/linear_interpolators_pocl.clh'
with open(src_file, 'r') as fid:
    src_content = fid.read()

prg = cl.Program(ctx, src_content).build()
knl_p2m_rectmesh3d = prg.p2m_rectmesh3d
knl_m2p_rectmesh3d = prg.m2p_rectmesh3d

prrr

knl = prg.sum  # Use this Kernel object for repeated calls
knl(queue, a.shape, None, a_dev.data, b_dev.data, dest_dev.data)
# The second argument gives the size od the computing grid (number of threads)
# See here: https://documen.tician.de/pyopencl/runtime_program.html#pyopencl.Kernel.__call__

# Here he compares the result with the operation done with the numpy syntax
print(la.norm((dest_dev - (a_dev+b_dev)).get()))