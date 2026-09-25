# Compatibility targets; Meson is the single native build definition.
PYTHON ?= python
.PHONY: all local cern f2py cythonize
all local cern f2py cythonize:
	$(PYTHON) tools/legacy_build.py
