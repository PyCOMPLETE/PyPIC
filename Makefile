PYTHON ?= python

.PHONY: all f2py
all: f2py

f2py:
	$(PYTHON) -m pip install -e .
