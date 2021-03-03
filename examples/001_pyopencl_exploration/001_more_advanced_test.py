#!/usr/bin/env python

import numpy as np

import pyopencl as cl
import pyopencl.array as cla

from gpyfft import GpyFFT_Error
from gpyfft import GpyFFT

from matplotlib import pyplot as plt

import sys

data = np.sin(0.1 * np.arange(2048, dtype=np.float32))

G = GpyFFT(debug=False)

context = cl.create_some_context()
queue = cl.CommandQueue(context)

dev_data = cla.zeros(queue, data.shape, np.float32)
dev_fft_data = cla.zeros(queue, (int(data.shape[0]/2) + 1,), np.complex64)

plan = G.create_plan(context, data.shape)
plan.inplace = False
plan.precision = 1   # Wish I could do gpyfft.CLFFT_SINGLE
plan.scale_forward = 1
plan.layouts = (5,3) # Wish ... (gpyfft.CLFFT_REAL, gpyfft.CLFFT_HERMITIAN_INTERLEAVED)

plan.bake(queue)

dev_data.set(data)

plan.enqueue_transform((queue,),
               (dev_data.data,),
               (dev_fft_data.data,))

queue.finish()

cl_fft_data = dev_fft_data.get()

np_fft = np.fft.rfft(data)
plt.subplot(311)
plt.plot(np.abs(cl_fft_data), label='OpenCL')
plt.plot(np.abs(np_fft), label='Numpy')
plt.legend()

plt.subplot(312)
plt.plot(np.abs(cl_fft_data) - np.abs(np_fft), label='Amp Diff')
plt.legend()

plt.subplot(313)
plt.plot(np.angle(cl_fft_data) - np.angle(np_fft), label='Phase Diff')
plt.legend()

plt.show()
