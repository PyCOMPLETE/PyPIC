"""Generate all three datasets consumed by 008b, in the working directory."""

import numpy as np
from scipy.io import savemat

from PyPIC.Bassetti_Erskine import Interpolated_Bassetti_Erskine
from PyPIC.FiniteDifferences_ShortleyWeller_SquareGrid import (
    FiniteDifferences_ShortleyWeller_SquareGrid,
)
from PyPIC.FiniteDifferences_Staircase_SquareGrid import (
    FiniteDifferences_Staircase_SquareGrid,
)
from PyPIC.geom_impact_ellip import ellip_cham_geom_object

x_aper, y_aper = 0.04, 0.02
Dh = 0.5e-3
sigmax, sigmay = 1e-3, 0.5e-3
chamber = ellip_cham_geom_object(x_aper, y_aper)
phase = np.linspace(0, 2 * np.pi, 1000)
theta = np.mod(np.arctan2(y_aper * np.sin(phase), x_aper * np.cos(phase)),
               2 * np.pi)
xmax_test_list = np.arange(1, 1601) * 0.025e-3

for name, solver in [
    ('FDSW', FiniteDifferences_ShortleyWeller_SquareGrid),
    ('FDSC', FiniteDifferences_Staircase_SquareGrid),
    ('BE', Interpolated_Bassetti_Erskine),
]:
    if name == 'BE':
        # The analytic solver already contains the field of a unit line charge.
        pic = solver(x_aper=x_aper, y_aper=y_aper, Dh=Dh,
                     sigmax=sigmax, sigmay=sigmay, n_imag_ellip=20)
    else:
        pic = solver(chamb=chamber, Dh=Dh, sparse_solver='scipy_slu')
        xx, yy = np.meshgrid(pic.xg, pic.yg, indexing='ij')
        rho = np.exp(-xx**2 / (2 * sigmax**2) - yy**2 / (2 * sigmay**2))
        rho /= 2 * np.pi * sigmax * sigmay
        pic.solve(rho=rho)

    fields = [pic.gather(radius * np.cos(phase),
                         radius * y_aper / x_aper * np.sin(phase))
              for radius in xmax_test_list]
    ex = np.array([field[0] for field in fields])
    ey = np.array([field[1] for field in fields])
    if not (np.all(np.isfinite(ex)) and np.all(np.isfinite(ey))):
        raise RuntimeError(f'{name} produced non-finite fields')
    filename = f'norepository_{name}_Dh{Dh * 1e3:.1f}mm.mat'
    savemat(filename, {'Ex': ex, 'Ey': ey, 'xmax_test_list': xmax_test_list,
                      'x_aper': x_aper, 'theta': theta}, oned_as='row')
    print(f'Wrote {filename}')
