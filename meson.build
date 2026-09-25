project('PyCOMPLETE-PyPIC', 'c', 'fortran', meson_version: '>=1.3')
py = import('python').find_installation(pure: false)
np_inc = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
f2py_inc = run_command(py, ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'], check: true).stdout().strip()
incs = include_directories(np_inc, f2py_inc, '.')
py.install_sources(['Bassetti_Erskine.py', 'FFT_OpenBoundary.py', 'FFT_OpenBoundary_SquareGrid.py', 'FFT_PEC_Boundary_SquareGrid.py', 'FiniteDifferences_ShortleyWeller_SquareGrid.py', 'FiniteDifferences_ShortleyWeller_SquareGrid_extrapolation.py', 'FiniteDifferences_Staircase_SquareGrid.py', 'MultiGrid.py', 'PyPIC_Scatter_Gather.py', '__init__.py', 'geom_impact_ellip.py', 'geom_impact_poly.py', 'myloadmat_to_obj.py', 'mystyle.py', 'simple_polygon.py'], subdir: 'PyPIC')

rhocompute_src = custom_target('rhocompute_wrappers',
  input: 'compute_rho.f', output: ['rhocomputemodule.c', 'rhocompute-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'rhocompute', '--lower'])
py.extension_module('rhocompute', ['compute_rho.f', rhocompute_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyPIC')

int_field_for_src = custom_target('int_field_for_wrappers',
  input: 'interp_field_for.f', output: ['int_field_formodule.c', 'int_field_for-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'int_field_for', '--lower'])
py.extension_module('int_field_for', ['interp_field_for.f', int_field_for_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyPIC')

errffor_src = custom_target('errffor_wrappers',
  input: 'errfff.f', output: ['errfformodule.c', 'errffor-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'errffor', '--lower'])
py.extension_module('errffor', ['errfff.f', errffor_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyPIC')

int_field_for_border_src = custom_target('int_field_for_border_wrappers',
  input: 'interp_field_for_with_border.f', output: ['int_field_for_bordermodule.c', 'int_field_for_border-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'int_field_for_border', '--lower'])
py.extension_module('int_field_for_border', ['interp_field_for_with_border.f', int_field_for_border_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyPIC')

meson.add_dist_script(py, files('tools/prepare_sdist.py'))
