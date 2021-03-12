all: f2py 

f2py:
	f2py -m rhocompute -c compute_rho.f
	f2py -m int_field_for -c interp_field_for.f
	f2py -m int_field_for_border -c interp_field_for_with_border.f
	f2py -m errffor -c errfff.f

cythonize:
	python setup.py build_ext -i
