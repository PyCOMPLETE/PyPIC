from . import platforms
from .platforms import cpu as _cpu
from .platforms import pocl as _pocl

platforms.XfCpuPlatform = _cpu.XfCpuPlatform
platforms.XfPoclPlatform = _pocl.XfPoclPlatform

from .fieldmaps.interpolated import TriLinearInterpolatedFieldMap

from .solvers.fftsolvers import FFTSolver3D

