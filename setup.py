import numpy as np

from setuptools import setup, Extension

from Cython.Distutils import build_ext
from Cython.Build import cythonize


cy_ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
cy_ext = [
    Extension(
        "CyFPPS",
        sources=[ # the Cython source and additional C++ source files generate and compile C++ code
            'FPPS/CyFPPS.pyx', 'FPPS/FPPSWrapper.cc',
            'FPPS/ChangeCoord.cc', 'FPPS/ElectricFieldSolver.cc', 'FPPS/Mesh.cc',
	    'FPPS/ChangeCoord_Frac.cc', 'FPPS/FastPolarPoissonSolver.cc',  'FPPS/NonLinearMesh.cc',
	    'FPPS/ChangeCoord_Tanh.cc', 'FPPS/PolarBeamRepresentation.cc',
	    'FPPS/ChargeDistribution.cc', 'FPPS/FunctionsFPPS.cc'],
        language="c++", include_dirs=[np.get_include()], libraries=['fftw3', 'm'])
]


setup(
    name='PyPIC',
    description='Collection of Python Particle-In-Cell solvers.',
    url='http://github.com/PyCOMPLETE/PyPIC',
    packages=['PyPIC'],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(cy_ext, **cy_ext_options),
    install_requires=[
        'numpy',
        'cython'
    ]
)
