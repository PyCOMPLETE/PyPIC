// 2nd order FD scheme for gradient with uniform mesh spacing
#include <stdio.h>

__constant__ int mx, my, mz;
__constant__ double dxInv, dyInv, dzInv;

__global__ void gradient_x(double* f, double* df) {
    //f: vector of function values
    //df: vector of derivatives (output)

    int i  = blockIdx.x*blockDim.x + threadIdx.x;
    int j  = blockIdx.y*blockDim.y + threadIdx.y;
    int k  = blockIdx.z;

    int globalIdx = k * mx * my + j * mx + i;
    if (i > 0 && i < mx-1) {
        df[globalIdx] = (-0.5*f[globalIdx-1] + 0.5*f[globalIdx+1])*dxInv;
    } else if (i == 0 && mx > 1) {
        df[globalIdx] = (-f[globalIdx] + f[globalIdx+1])*dxInv;
    } else if (i == mx-1 && mx > 1) {
        df[globalIdx] = (-f[globalIdx-1] + f[globalIdx])*dxInv;
    }
}

__global__ void gradient_y(double* f, double* df) {
    //f: vector of function values
    //df: vector of derivatives (output)

    int i  = blockIdx.x*blockDim.x + threadIdx.x;
    int j  = blockIdx.y*blockDim.y + threadIdx.y;
    int k  = blockIdx.z;

    int globalIdx = k * mx * my + j * mx + i;
    if (j > 0 && j < my-1) {
        df[globalIdx] = (-0.5*f[globalIdx-mx] + 0.5*f[globalIdx+mx])*dyInv;
    } else if (j == 0 && my > 1) {
        df[globalIdx] = (-f[globalIdx] + f[globalIdx+mx])*dyInv;
    } else if (j == my-1 && my > 1) {
        df[globalIdx] = (-f[globalIdx-mx] + f[globalIdx])*dyInv;
    }
}
__global__ void gradient_z(double* f, double* df) {
    //f: vector of function values
    //df: vector of derivatives (output)

    int i  = blockIdx.x*blockDim.x + threadIdx.x;
    int j  = blockIdx.y*blockDim.y + threadIdx.y;
    int k  = blockIdx.z;

    int globalIdx = k * mx * my + j * mx + i;
    if (k > 0 && k < mz-1) {
        df[globalIdx] = (-0.5*f[globalIdx-mx*my] + 0.5*f[globalIdx+mx*my])*dzInv;
    } else if (k == 0 && mz > 1) {
        df[globalIdx] = (-f[globalIdx] + f[globalIdx+mx*my])*dzInv;
    } else if (k == mz-1 && mz > 1) {
        df[globalIdx] = (-f[globalIdx-mx*my] + f[globalIdx])*dzInv;
    }
}
