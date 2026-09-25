# PyPIC

Install the standard CPU package with `python -m pip install PyCOMPLETE-PyPIC`.
The distribution is named `PyCOMPLETE-PyPIC`; the import remains `PyPIC`.
Python 3.10–3.14 and NumPy 2.x are the target support matrix.
Linux x86_64 and macOS Intel/Apple Silicon wheels avoid local compilation.

PyKLU is a required dependency and is installed automatically.
The unrelated PyPI project `pypic` is not this package. Use a clean environment
and do not install both distributions into the same environment.
The GPU and FFTW polar backends are not included. Standard FFT solvers use
NumPy, with optional pyFFTW acceleration if separately installed.

## Build from source

Install a C compiler and GNU Fortran (`gcc gfortran` on Linux; Xcode command
line tools and Homebrew `gcc` on macOS). Then run `python -m pip install .`.
Build requirements are installed automatically in an isolated environment.
On macOS set `FC` to the installed versioned gfortran executable if necessary.

For development:

```sh
python -m pip install meson-python meson ninja 'numpy>=2,<3' 'Cython>=3'
python -m pip install --no-build-isolation -e '.[test]'
```

Keep that environment's build dependencies installed: editable imports can
rebuild compiled extensions. Legacy scripts remain available but are not
used by pip. Build release artifacts with `python -m build`.

Run installed-package tests from outside the checkout:
`python /path/to/checkout/tools/test_installed.py`.
See `PACKAGING.md` for release and validation details.

The package version is defined only in `[project].version` in `pyproject.toml`.
`__version__` reads installed distribution metadata, including editable installs.
After changing the version, reinstall the package to refresh that metadata.
Legacy uninstalled source checkouts report `unknown (not installed)`.
