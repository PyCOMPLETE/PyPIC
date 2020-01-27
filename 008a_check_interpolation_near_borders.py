import sys
sys.path.append('..')

import numpy as np
from PyPIC.geom_impact_ellip import ellip_cham_geom_object

x_aper = .04
y_aper = .02

Dh = .5e-3
sigmax=1e-3
sigmay=.5e-3

chamber = ellip_cham_geom_object(x_aper, y_aper)

import PyPIC.FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
pic = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh)
filename_out = 'norepository_FDSW_Dh%.1fmm.mat'%(Dh*1e3)

import PyPIC.FiniteDifferences_Staircase_SquareGrid as PIC_FD
pic = PIC_FD.FiniteDifferences_Staircase_SquareGrid(chamb = chamber, Dh = Dh)
filename_out = 'norepository_FDSWextrap_Dh%.1fmm.mat'%(Dh*1e3)

# import PyPIC.Bassetti_Erskine as BE
# pic = BE.Interpolated_Bassetti_Erskine(
#         x_aper=chamber.x_aper, y_aper=chamber.y_aper, Dh=Dh, 
#         sigmax=sigmax, sigmay=sigmay, n_imag_ellip=20)
# filename_out = 'norepository_BE_Dh%.1fmm.mat'%(Dh*1e3)



N_test =1000
thp = np.linspace(0, 2*np.pi, N_test)
x_bou = x_aper*np.cos(thp)
y_bou = y_aper*np.sin(thp)
theta = np.arctan2(y_bou, x_bou)
theta[theta<0] = theta[theta<0]+2*np.pi

dec_fact = 1
err_abs_list = []
err_rel_list = []

xmax_test_list = np.arange(.025e-3, x_aper+.025e-3, .025e-3)



YY,XX = np.meshgrid(pic.yg, pic.xg)
rho_mat=1./(2.*np.pi*sigmax*sigmay)*np.exp(-(XX)**2/(2.*sigmax**2)-(YY)**2/(2.*sigmay**2))
try:
    pic.solve(rho = rho_mat)
except ValueError as err:
    print('Got ValueError:', err)
    
    
Ex_list = []
Ey_list = []
for xmax_test in xmax_test_list: 
    
    sc_test = xmax_test/x_aper
    x_test = sc_test*x_aper*np.cos(thp)
    y_test = sc_test*y_aper*np.sin(thp)

    Ex, Ey = pic.gather(x_test, y_test)

    Ex_list.append(Ex)
    Ey_list.append(Ey)
    
Ex_list = np.array(Ex_list)
Ey_list = np.array(Ey_list)

import scipy.io as sio
sio.savemat(filename_out,{\
    'Ex':Ex_list,
    'Ey':Ey_list,
    'xmax_test_list':xmax_test_list,
    'x_aper':x_aper, 'theta':theta}, oned_as='row')
