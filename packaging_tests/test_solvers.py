import numpy as np
import pytest
from numpy.testing import assert_allclose
from PyPIC.geom_impact_ellip import ellip_cham_geom_object
from PyPIC.FiniteDifferences_ShortleyWeller_SquareGrid import FiniteDifferences_ShortleyWeller_SquareGrid as SW
from PyPIC.FiniteDifferences_Staircase_SquareGrid import FiniteDifferences_Staircase_SquareGrid as Staircase


def solve(pic, scale=1.):
    pic.scatter(np.array([0.]), np.array([0.]), np.array([1.e5 * scale]))
    pic.solve()
    return np.array(pic.gather(np.array([.003, -.003]), np.zeros(2)))


@pytest.mark.parametrize('cls', [SW, Staircase])
def test_solver_linearity_and_symmetry(cls):
    pic = cls(chamb=ellip_cham_geom_object(.02, .02), Dh=.002, sparse_solver='scipy_slu')
    field = solve(pic)
    assert np.isfinite(field).all()
    assert abs(field[0, 0]) > 0
    assert_allclose(field[0, 0], -field[0, 1], rtol=1e-10)
    assert_allclose(solve(pic, 2.), field * 2., rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('cls', [SW, Staircase])
def test_klu_really_used(cls):
    klu = pytest.importorskip('PyKLU')
    chamber = ellip_cham_geom_object(.02, .02)
    pic = cls(chamb=chamber, Dh=.002, sparse_solver='PyKLU')
    assert isinstance(pic.luobj, klu.Klu), 'KLU silently fell back to SciPy'
    reference = cls(chamb=chamber, Dh=.002, sparse_solver='scipy_slu')
    assert_allclose(solve(pic), solve(reference), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('kind', ['open', 'pec'])
def test_fft_solver(kind):
    if kind == 'open':
        from PyPIC.FFT_OpenBoundary import FFT_OpenBoundary as FFT
    else:
        from PyPIC.FFT_PEC_Boundary_SquareGrid import FFT_PEC_Boundary_SquareGrid as FFT
    pic = FFT(x_aper=.02, y_aper=.02, Dh=.002, fftlib='numpy')
    field = solve(pic)
    assert np.isfinite(field).all()
    assert abs(field[0, 0]) > 0
    assert_allclose(solve(pic, 2.), 2. * field, rtol=1e-10, atol=1e-10)


def test_border_interpolation():
    from PyPIC.int_field_for_border import int_field_border
    ex, ey = int_field_border([.25], [.5], 0., 0., 1., 1., np.ones((3, 3)),
        np.full((3, 3), 2.), np.ones((3, 3), dtype=np.int8))
    assert_allclose(ex, [1.])
    assert_allclose(ey, [2.])


def test_multigrid():
    from PyPIC.MultiGrid import AddTelescopicGrids
    pic = SW(chamb=ellip_cham_geom_object(.02, .02), Dh=.002, sparse_solver='scipy_slu')
    multi = AddTelescopicGrids(pic_main=pic, f_telescope=.5,
        target_grid=dict(x_min_target=-.004, x_max_target=.004, y_min_target=-.004,
                         y_max_target=.004, Dh_target=.001),
        N_nodes_discard=1, N_min_Dh_main=2, sparse_solver='scipy_slu')
    field = solve(multi)
    assert np.isfinite(field).all()
    assert abs(field[0, 0]) > 0
    assert_allclose(solve(multi, 2.), 2. * field, rtol=1e-9, atol=1e-9)
