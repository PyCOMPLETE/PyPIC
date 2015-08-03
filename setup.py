#!/usr/bin/python

import numpy as np

import re, os, sys, subprocess
import numpy as np

from setuptools import setup, Extension

from numpy.distutils.core import setup, Extension

args = sys.argv[1:]
# Make a `cleanall` rule to get rid of intermediate and library files
if "cleanall" in args:
    print "Deleting shared libraries and"
    # Just in case the build directory was created by accident,
    # note that shell=True should be OK here because the command is constant.
    subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf *.so.dSYM", shell=True, executable="/bin/bash")
    subprocess.Popen("find ./ -name *.so | xargs rm", shell=True)

    # Now do a normal clean
    sys.argv[1] = "clean"
    exit(1)


# We want to always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext") + 1, "--inplace")

verstr = 1.1

setup(
    name='PyPIC',
    version=verstr,
    description='PyPIC: Particle in Cell codes',
    url='http://github.com/PyCOMPLETE/PyPIC',
    packages=['PyPIC'],
    install_requires=[
        'numpy',
        'scipy',
        'cython',
    ],
    ext_modules = [
        Extension('rhocompute', ['p2m/compute_rho.f']),
        Extension('int_field_for', ['m2p/interp_field_for.f']),
        Extension('int_field_for_border', ['m2p/interp_field_for_with_border.f']),
        Extension('vectsum', ['vectsum.f']),
        ]
    )
