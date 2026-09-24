import importlib
import numpy as np
import pytest
from numpy.testing import assert_allclose

PACKAGE = "PyPIC"


def test_charge_deposition_and_interpolation():
    rho_mod = importlib.import_module(PACKAGE + '.rhocompute')
    field_mod = importlib.import_module(PACKAGE + '.int_field_for')
    rho = rho_mod.compute_sc_rho([0.25], [0.5], [8.], 0., 0., 1., 1., 3, 3)
    assert_allclose(rho[:2, :2], [[3., 3.], [1., 1.]])
    assert_allclose(rho.sum(), 8.)
    ex, ey = field_mod.int_field([0.25], [0.5], 0., 0., 1., 1., np.ones((3, 3)), np.full((3, 3), 2.))
    assert_allclose(ex, [1.])
    assert_allclose(ey, [2.])


def test_complex_error_function():
    from scipy.special import wofz
    mod = importlib.import_module(PACKAGE + '.errffor')
    real, imag = mod.errf(0.3, 0.4)
    assert_allclose(real + 1j * imag, wofz(0.3 + 0.4j), rtol=1e-6)


def test_version_and_installed_location():
    from importlib.metadata import version
    package = importlib.import_module(PACKAGE)
    assert package.__version__ == version('PyCOMPLETE-PyPIC')


def test_public_module_imports():
    import pkgutil
    package = importlib.import_module(PACKAGE)
    for module in pkgutil.iter_modules(package.__path__):
        if module.name == 'Transverse_Efield_map_for_frozen_cloud':
            continue  # optional PyHEADTAIL integration is tested separately
        importlib.import_module(PACKAGE + '.' + module.name)
