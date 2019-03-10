import sys, os
BIN=os.path.expanduser('../')
sys.path.append(BIN)
BIN=os.path.expanduser('../PyHEADTAIL/testing/script-tests/')
sys.path.append(BIN)
import PyPIC.FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
import PyPIC.FiniteDifferences_Staircase_SquareGrid as PIC_FD
import PyPIC.geom_impact_ellip as ell
import pylab as pl
import numpy as np
from scipy import rand
from scipy.constants import e, epsilon_0
from PyPIC.MultiGrid import AddInternalGrid

def analytic_solution(x_probes, y_probes, x_part, y_part, nel_part, charge):
    Ex_probes = []
    Ey_probes = []
    
    for x_probe, y_probe in zip(x_probes, y_probes):
        r_probe = np.sqrt(x_probe**2 + y_probe**2)
        q_inside = np.sum(nel_part[x_part**2+ y_part**2 < r_probe**2])*charge
        Er_probe = q_inside/(epsilon_0*2*np.pi*r_probe)
        Ex_probes.append(Er_probe  * x_probe/r_probe)
        Ey_probes.append(Er_probe  * y_probe/r_probe)
    
    return Ex_probes, Ey_probes

qe = e
eps0 = epsilon_0

#chamber parameters
x_aper = 25e-3
y_aper = 25e-3
Dh_main = .1e-3

#traditional settings
#~ tol_der = 0.01
#~ tol_stem = 0.01

#new default settings
tol_der = 0.1
tol_stem = 0.01

#build chamber
chamber = ell.ellip_cham_geom_object(x_aper = x_aper, y_aper = y_aper)

# build main pic
pic_SW = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, 
        Dh = Dh_main, tol_stem = tol_stem, tol_der = tol_der)


#~ # generate beam
N_part = 100000
r_charge=4e-3
x_part = r_charge*(2.*rand(N_part)-1.)
y_part = r_charge*(2.*rand(N_part)-1.)
mask_keep  = x_part**2+y_part**2<r_charge**2
x_part = x_part[mask_keep]
y_part = y_part[mask_keep]
nel_part = 0*x_part+1.

#pic scatter
pic_SW.scatter(x_part, y_part, nel_part, charge=qe)
pic_SW.solve()


theta=np.linspace(0, 2*np.pi, 1000)
r_probes=x_aper*0.999999
x_probes = r_probes*np.cos(theta)
y_probes = r_probes*np.sin(theta)


#~ #pic gather
Ex_probes, Ey_probes = pic_SW.gather(x_probes, y_probes)

#analytic formula
E_r_th_x, E_r_th_y = analytic_solution(x_probes, y_probes, x_part, y_part, nel_part, qe)

pl.close('all')

#plot fields
pl.figure(1)
pl.plot(theta*180/np.pi, Ex_probes, label = 'FD ShorleyWeller')
pl.plot(theta*180/np.pi, E_r_th_x, label = 'Analytic')
pl.legend()
pl.ylabel('Ex [V/m]')
pl.xlabel('theta [deg]')

pl.figure(2)
pl.plot(theta*180/np.pi, Ey_probes, label = 'FD ShorleyWeller')
pl.plot(theta*180/np.pi, E_r_th_y, label = 'Analytic')
pl.legend()
pl.ylabel('Ey [V/m]')
pl.xlabel('theta [deg]')





pl.show()


    
