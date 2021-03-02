from setuptools import setup, find_packages, Extension

#######################################
# Prepare list of compiled extensions #
#######################################

extensions = []

# C extension called via cython
from Cython.Build import cythonize
cython_extensions = [
        Extension(
            name='xfields.fieldmaps.linear_interpolators',
            sources=['xfields/csrc/p2m_cpu.pyx'],
            include_dirs=['xfields/csrc'],
        ),
        # Other cython extensions can be added here
    ]
# Cython extensions need to be cythonized before being added to the main
# extension list:
extensions += cythonize(cython_extensions)

#########
# Setup #
#########

setup(
    name='xfields',
    version='0.0.0',
    description='Field Maps and Particle In Cell',
    url='https://github.com/giadarol/xfields',
    author='Giovanni Iadarola',
    packages=find_packages(),
    ext_modules = extensions,
    install_requires=[
        'numpy>=1.0',
        ]
    )
