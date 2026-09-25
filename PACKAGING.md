# Packaging and release notes

## Source baseline

This branch starts at PyCOMPLETE/PyPIC revision
`2b011d9`. Existing fork checkouts are preserved.
The inspected ekatralis/PyPIC fork has no differences from this upstream
revision. KLU imports now use the published PyKLU API.

## Supported build targets

CI builds CPython 3.10, 3.11, 3.12, 3.13, and 3.14 for manylinux_2_28 x86_64,
macOS 14+ x86_64, and macOS 14+ arm64. Free-threaded interpreters, Windows,
Linux ARM, GPU backends and the polar FFTW solver are excluded. cibuildwheel
repairs platform wheels with auditwheel/delocate, including needed Fortran
runtime libraries. No compiler is required to install matching wheels.
NumPy 2.x is required. pip resolves Python-compatible releases of dependencies.
A CI configuration is not evidence of a passing platform: inspect all jobs
before advertising a release matrix.

## Local development and validation

Use `python -m pip install .` for an isolated source build. Install PyPIC first
when testing unpublished local PyECLOUD changes. Install build dependencies
before `python -m pip install --no-build-isolation -e .` for editable builds.
Run `python tools/test_installed.py` to copy tests to a temporary directory and
avoid importing the source tree accidentally. PyKLU is a required dependency of both packages. Tests assert the real solver
is used and fail when PyKLU is absent; CI obtains it from package dependencies.

`python -m build` creates an sdist then builds its wheel in isolation. Meson
archives committed files only: commit changes before validating the source
archive. `python -m twine check dist/*` validates distribution metadata.
The source archive includes small packaging test fixtures, native sources and
headers. Large legacy simulation/reference datasets remain in the Git repo,
not release archives. Neither tests nor examples are installed in wheels.

## Release sequence

1. Resolve upstream publication rights and project ownership. PyPIC's existing
   source notices explicitly require redistribution permission; COPYING retains
   the notice rather than assigning a new license.
2. Confirm availability/ownership of `PyCOMPLETE-PyPIC` and `PyECLOUD` on both
   PyPI and TestPyPI. The unrelated `pypic` distribution must not be installed
   alongside PyCOMPLETE-PyPIC, as both may own the same import namespace.
3. Set the release version only in `[project].version` in pyproject.toml.
   Runtime `__version__` reads installed metadata; reinstall after version changes. Configure trusted publishers for `.github/workflows/release.yml`
   and protected `testpypi`/`pypi` environments in each upstream repository.
4. Run the build matrix and review tests. Stage PyPIC first; then PyECLOUD.
   PyECLOUD CI requires its PyPIC dependency to be available to pip. For local
   staging use a wheelhouse containing both packages (`--find-links`).
5. Use the manual Release workflow targeting TestPyPI. Download the exact
   staged versions with `pip download --index-url https://test.pypi.org/simple/
   --no-deps`, then install those files in a fresh environment, resolving normal
   third-party dependencies from PyPI. Verify that a normal installation includes PyKLU.
6. Verify optional tracking independently using the manual tracking workflow.
   Published PyHEADTAIL currently builds from source; do not claim successful
   tracking on a Python/platform combination until that job passes.
7. Publish PyPIC to PyPI before PyECLOUD, using the same reviewed commit and
   version. Verify a fresh `pip install PyECLOUD`, including actual KLU execution.

The manual release workflow does not run merely because a branch is pushed.
No artifacts have been uploaded by the local packaging implementation.

## Local validation completed (2026-09-24)

Linux wheel builds and installed tests pass on CPython 3.10–3.14 (12 PyPIC and
14 PyECLOUD tests per version). Isolated sdist-to-wheel builds, metadata,
editable installs, compiler-free wheel installs, KLU execution, and declared
minimum dependencies passed. Published PyHEADTAIL installation and tracking
imports also passed on all five Linux Python versions.
MacOS and release workflows have not been run. Local wheels require glibc 2.31;
CI targets manylinux_2_28 using its older build image. Nothing was published.

## Metadata update (2026-09-25)

PyKLU>=0.2.0 is now a direct runtime dependency of both distributions; the
`klu` extra was removed. Existing solver defaults and the SciPy fallback are
unchanged. PyHEADTAIL remains optional via `PyECLOUD[tracking]`.
`pyproject.toml` is the only version source. Meson does not duplicate the
version, `_version.py` was removed, and runtime version reporting reads
installed metadata. Uninstalled legacy source imports report an unknown
version rather than maintaining a second version literal.

## Package directory layout

Runtime Python modules live in `PyPIC/`; native sources and declarations live
in `PyPIC/_native/`. Meson installs the package directory without enumerating
Python files. Native sources stay in the source archive; compiled extensions
are installed in `PyPIC/`. Add new runtime Python modules directly to the
package directory. Build tools, tests and standalone scripts stay at the root.
Optional PyPIC GPU/FPPS directories remain outside the shipped CPU package.

Legacy `make` targets and `python setup.py build_ext -i` now delegate to the
same editable pip build. Install the documented editable-build dependencies
first. These commands no longer create extension files inside the source
package. PyECLOUD's `setup_pyecloud` and `cythonize` use the same path.
