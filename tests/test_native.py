"""Exercise installed native modules without requiring plotting."""

from importlib import import_module

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import wofz

from PyPIC import errffor
from PyPIC.PyPIC_Scatter_Gather import PyPIC_Scatter_Gather


@pytest.mark.parametrize('name', [
    'PyPIC.MultiGrid', 'PyPIC.Bassetti_Erskine', 'PyPIC.FFT_OpenBoundary',
    'PyPIC.FFT_PEC_Boundary_SquareGrid',
])
def test_public_modules_import(name):
    import_module(name)


def test_scatter_conserves_charge_and_gather_reproduces_linear_field():
    pic = PyPIC_Scatter_Gather(xg=np.arange(5.), yg=np.arange(6.) * 0.5)
    x = np.array([0.25, 1.5, 2.75])
    y = np.array([0.125, 0.75, 1.625])
    weights = np.array([1., 2., 3.])
    pic.scatter(x, y, weights, charge=2.)
    assert_allclose(pic.rho.sum() * pic.dx * pic.dy, 2 * weights.sum())
    assert_allclose(pic.rho[0, 0] * pic.dx * pic.dy, 2 * 0.75 * 0.75)

    xx, yy = np.meshgrid(pic.xg, pic.yg, indexing='ij')
    pic.efx = 2 * xx + 3 * yy
    pic.efy = xx - yy
    ex, ey = pic.gather(x, y)
    assert_allclose(ex, 2 * x + 3 * y)
    assert_allclose(ey, x - y)


def test_border_interpolation_renormalizes_weights():
    from PyPIC import int_field_for_border as module
    inside = np.ones((4, 4), dtype=np.int32)
    inside[0, 0] = 0
    field = np.full((4, 4), 7.)
    field[0, 0] = 1000.  # The outside node must make no contribution.
    ex, ey = module.int_field_border(
        np.array([0.25]), np.array([0.25]), 0., 0., 1., 1.,
        field, -field, inside)
    assert_allclose(ex, 7.)
    assert_allclose(ey, -7.)


@pytest.mark.parametrize('z', [0j, 0.3 + 0.7j, 2. + 1.j, 10. + 2.j])
def test_complex_error_function(z):
    real, imag = errffor.errf(z.real, z.imag)
    assert_allclose(real + 1j * imag, wofz(z), rtol=2e-6, atol=1e-10)


def test_poisson_solve_satisfies_discrete_equation():
    from scipy.constants import epsilon_0
    from PyPIC.FiniteDifferences_Staircase_SquareGrid import (
        FiniteDifferences_Staircase_SquareGrid,
    )
    from PyPIC.geom_impact_ellip import ellip_cham_geom_object

    pic = FiniteDifferences_Staircase_SquareGrid(
        ellip_cham_geom_object(1., 1.), Dh=0.1, sparse_solver='scipy_slu')
    pic.scatter(np.array([0.15]), np.array([0.05]), np.array([1.]))
    pic.solve()
    phi = pic.phi
    laplacian = (phi[2:, 1:-1] + phi[:-2, 1:-1]
                 + phi[1:-1, 2:] + phi[1:-1, :-2]
                 - 4 * phi[1:-1, 1:-1]) / pic.Dh**2
    xx, yy = np.meshgrid(pic.xg[1:-1], pic.yg[1:-1], indexing='ij')
    interior = xx**2 + yy**2 < 0.8**2
    expected = -pic.rho[1:-1, 1:-1] / epsilon_0
    assert np.max(np.abs(expected)) > 0
    assert_allclose(laplacian[interior], expected[interior],
                    atol=np.max(np.abs(expected)) * 1e-12)
    assert np.all(np.isfinite(pic.gather(np.array([0.2]), np.array([0.1]))))
