"""Compare NumPy and pyFFTW on the same forward/product/inverse FFT operation."""

import argparse
from time import perf_counter

import numpy as np
from PyPIC.FFT_OpenBoundary_SquareGrid import FFT_OpenBoundary_SquareGrid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repetitions', type=int, default=100)
args = parser.parse_args()
if args.repetitions < 1:
    parser.error('--repetitions must be positive')

try:
    import pyfftw
except ImportError as exc:
    raise SystemExit("This benchmark requires pyfftw: pip install 'PyPIC[fftw]'") from exc

pic = FFT_OpenBoundary_SquareGrid(x_aper=6., y_aper=7., Dh=0.1, fftlib='numpy')
data = pic.fgreen.copy()
fftobj = pyfftw.builders.fft2(data.copy())
ifftobj = pyfftw.builders.ifft2(fftobj(data).copy())

# Warm up both backends; exclude FFTW planning from the timings.
np.fft.ifft2(np.fft.fft2(data) * data)
ifftobj(fftobj(data) * data)

start = perf_counter()
for _ in range(args.repetitions):
    transf = np.fft.fft2(data)
    itransf = np.fft.ifft2(transf * data)
t_numpy = (perf_counter() - start) / args.repetitions

start = perf_counter()
for _ in range(args.repetitions):
    transfw = fftobj(data)
    itransfw = ifftobj(transfw * data)
t_fftw = (perf_counter() - start) / args.repetitions

np.testing.assert_allclose(itransfw, itransf, rtol=1e-11, atol=1e-10)
print(f'NumPy: {1e3 * t_numpy:.3f} ms per iteration')
print(f'pyFFTW: {1e3 * t_fftw:.3f} ms per iteration')
