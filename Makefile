all:
	f2py -m rhocompute -c compute_rho.f
	f2py -m int_field_for -c interp_field_for.f
	f2py -m vectsum -c vectsum.f

