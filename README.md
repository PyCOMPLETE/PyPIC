# PyPIC

Particle-in-cell solvers for beam simulations. The repository contains the
`PyPIC` Python package, native Fortran sources, tests, and examples.
Existing imports such as `from PyPIC.MultiGrid import AddTelescopicGrids` remain
unchanged.

## Installation

Python 3.11 or newer and working C and Fortran compilers are required. For example,
in a conda environment install `c-compiler` and `fortran-compiler` from conda-forge,
then activate that environment so its compiler settings are available.

From the directory containing this checkout:

```sh
python -m pip install -e ./PyPIC
```

Or, from inside the checkout:

```sh
python -m pip install -e .
```

Pip installs Python build dependencies in an isolated build environment and
compiles all four core Fortran extensions. No manual `make` step or precompiled
binaries are needed. NumPy and SciPy are runtime dependencies.
The legacy GPU and FPPS solvers are no longer included.

Python edits take effect immediately. After changing Fortran
sources, rerun the installation command to rebuild the extensions. Use
`python -m pip install .` for a regular installation.

### Optional runtime dependencies

- `python -m pip install -e '.[fftw]'` installs the Python FFTW interface used by
  FFT solvers.
- `python -m pip install -e '.[examples]'` adds matplotlib for the example scripts.
  Some legacy examples need additional packages or modernization.

## Development and validation

```sh
python -m pip install -e '.[tests]'
python -m pytest
python -m pip install build
python -m build
```

The tests check native-module imports, numerical deposition/interpolation,
the complex error function, and a Poisson solve.
`python -m build` creates a source distribution and builds a wheel from it.
Generated binaries are specific to the Python/platform used for the build.

Legacy CPU plotting scripts are under `examples/cpu/`; automated tests
are under `tests/`. The root Makefile is only a convenience wrapper around pip.
Package version metadata lives in `PyPIC/_version.py`.
