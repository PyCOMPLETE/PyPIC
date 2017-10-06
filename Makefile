
.PHONY: all python2 python3 cythonize_2 cythonize_3

# On Arch Linux, the default python version is python3
# Therefore, the command f2py2 has to be used for python2 programs such as PyECLOUD.
# Ubuntu and Red Hat Linux use the standard convention where the default python version
# is python2.
ARCH_LINUX := $(shell grep "Arch Linux" /etc/os-release 2>/dev/null)

ifdef ARCH_LINUX
		F2PY_py2 = f2py2
		F2PY_py3 = f2py
else
		F2PY_py2 = f2py
		F2PY_py3 = f2py3
endif


all: python2

python2:
	$(F2PY_py2) -m rhocompute -c compute_rho.f
	$(F2PY_py2) -m int_field_for -c interp_field_for.f
	$(F2PY_py2) -m int_field_for_border -c interp_field_for_with_border.f
	$(F2PY_py2) -m vectsum -c vectsum.f
	$(F2PY_py2) -m errffor -c errfff.f

python3:
	$(F2PY_py3) -m rhocompute -c compute_rho.f
	$(F2PY_py3) -m int_field_for -c interp_field_for.f
	$(F2PY_py3) -m int_field_for_border -c interp_field_for_with_border.f
	$(F2PY_py3) -m vectsum -c vectsum.f
	$(F2PY_py3) -m errffor -c errfff.f

cythonize_2:
	python2 setup.py build_ext -i

cythonize_3:
	python3 setup.py build_ext -i

