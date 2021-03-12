from . import platforms
from .platforms import cpu as _cpu
from .platforms import pocl as _pocl
from .platforms import cu_py as _cu_py

platforms.XfCpuPlatform = _cpu.XfCpuPlatform
platforms.XfPoclPlatform = _pocl.XfPoclPlatform
platforms.XfCupyPlatform = _cu_py.XfCupyPlatform

from .fieldmaps.interpolated import TriLinearInterpolatedFieldMap

from .solvers.fftsolvers import FFTSolver3D

from .beam_elements.spacecharge import SpaceCharge3D
