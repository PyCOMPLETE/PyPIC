import numpy as np
import cupy as cp

#######################
# For a single kernel #
#######################

add_kernel = cp.RawKernel(r'''
extern "C" __global__
void my_add(const float* x1, const float* x2, float* y) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    y[tid] = x1[tid] + x2[tid];
}
''', 'my_add')
x1 = cp.array(np.arange(25, dtype=np.float32).reshape(5, 5))
x2 = cp.array(np.arange(25, dtype=np.float32).reshape(5, 5))
y = cp.zeros((5, 5), dtype=cp.float32)
add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments

#####################
# For a full module #
#####################

loaded_from_source = r'''
extern "C"{

__global__ void test_sum(const float* x1, const float* x2, float* y, \
                         unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        y[tid] = x1[tid] + x2[tid];
    }
}

__global__ void test_multiply(const float* x1, const float* x2, float* y, \
                              unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        y[tid] = x1[tid] * x2[tid];
    }
}

}'''
module = cp.RawModule(code=loaded_from_source)
ker_sum = module.get_function('test_sum')
ker_times = module.get_function('test_multiply')
N = 10
x1 = cp.arange(N**2, dtype=cp.float32).reshape(N, N)
x2 = cp.ones((N, N), dtype=cp.float32)
y = cp.zeros((N, N), dtype=cp.float32)
ker_sum((N,), (N,), (x1, x2, y, N**2))   # y = x1 + x2
assert cp.allclose(y, x1 + x2)
ker_times((N,), (N,), (x1, x2, y, N**2)) # y = x1 * x2
assert cp.allclose(y, x1 * x2)

