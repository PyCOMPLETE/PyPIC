from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize(Extension(
           "CyFPPS",                                # the extesion name
           sources=['FPPS/CyFPPS.pyx', 'FPPS/FPPSWrapper.cc',
           'FPPS/ChangeCoord.cc', 'FPPS/ElectricFieldSolver.cc', 'FPPS/Mesh.cc',
		   'FPPS/ChangeCoord_Frac.cc', 'FPPS/FastPolarPoissonSolver.cc',  'FPPS/NonLinearMesh.cc',
		   'FPPS/ChangeCoord_Tanh.cc', 'FPPS/PolarBeamRepresentation.cc',
		   'FPPS/ChargeDistribution.cc', 'FPPS/FunctionsFPPS.cc'], 				  # the Cython source and additional C++ source files generate and compile C++ code
           language="c++",  libraries=[	'fftw3']                      
      )))
