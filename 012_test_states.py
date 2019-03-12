import numpy as np
import pylab as pl
from scipy.constants import e as qe
import PyPIC.mystyle as ms

# build chamber
x_aper = 2e-2; y_aper=1e-2
import geom_impact_ellip as ell
chamber = ell.ellip_cham_geom_object(x_aper = x_aper, y_aper = y_aper)

#build particle distribution
from scipy import randn
N_part = 10000; sigmax=.5e-3; sigmay=1e-3
x_mp = sigmax*randn(N_part);
y_mp = sigmay*randn(N_part);
nel_mp = x_mp*0.+1.

#build probes
N_probes = 100
n_sigma_probes = 1.
theta_probes = np.linspace(0, 2*np.pi, N_probes)
x_probes = n_sigma_probes*sigmax*np.cos(theta_probes)
y_probes = n_sigma_probes*sigmay*np.sin(theta_probes)

# build pics
pic_list = []

# Finite Difference Shortley-Weller
Dh=1e-3
import FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
pic_list.append(PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh, sparse_solver = 'PyKLU'))

# Finite Difference Staircase
Dh=1e-3
import FiniteDifferences_Staircase_SquareGrid as PIC_FDSC
pic_list.append(PIC_FDSC.FiniteDifferences_Staircase_SquareGrid(chamb = chamber, Dh = Dh, sparse_solver = 'PyKLU'))

#  Multi grid 
Sx_target = 5*sigmax
Sy_target = 5*sigmay
Dh_target = 0.1*min([sigmax, sigmay])
Dh_single = .5e-3
pic_singlegrid = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh_single, sparse_solver = 'PyKLU')
from MultiGrid import AddTelescopicGrids
pic_list.append(AddTelescopicGrids(pic_main = pic_singlegrid, f_telescope = 0.3, 
    target_grid = {'x_min_target':-Sx_target/2., 'x_max_target':Sx_target/2.,'y_min_target':-Sy_target/2.,'y_max_target':Sy_target/2.,'Dh_target':Dh_target}, 
    N_nodes_discard = 3., N_min_Dh_main = 10, sparse_solver='PyKLU'))

# test:

pl.close('all')
ms.mystyle_arial(fontsz = 14)

for i_pic, pic in enumerate(pic_list):
    
    # standard solve and gather
    pic.scatter(x_mp, y_mp, nel_mp, charge = qe)
    pic.solve()
    Ex_probes, Ey_probes = pic.gather(x_probes, y_probes)

    # build state list
    fact_states = np.linspace(1.5, 2., 4)
    N_states = len(fact_states)

    list_states = []
    for _ in fact_states:
        list_states.append(pic.get_state_object())
        
    for i_state, state in enumerate(list_states):
        state.scatter(x_mp, y_mp, nel_mp*fact_states[i_state], charge = qe)
     
    # solve states
    pic.solve_states(list_states)

    # gather and plot
    pl.figure(1+i_pic, figsize=(10, 6)).patch.set_facecolor('w')
    sp1 = pl.subplot(2,1,1)
    pl.plot(theta_probes, Ex_probes)
    sp2 = pl.subplot(2,1,2)
    pl.plot(theta_probes, Ey_probes)

    for i_state, state in enumerate(list_states):
        colorcurr = ms.colorprog(i_state, N_states)
        Ex_prb_state, Ey_prb_state = state.gather(x_probes, y_probes)
        pl.subplot(2,1,1)
        pl.plot(theta_probes, Ex_prb_state, '.', color=colorcurr, label = 'State %d'%i_state)
        pl.plot(theta_probes, Ex_probes*fact_states[i_state], '-', color=colorcurr, label = 'Ref. %d'%i_state)
        pl.xlabel('theta [deg]')
        pl.ylabel('Ex [V/m]')
        pl.subplot(2,1,2)
        pl.plot(theta_probes, Ey_prb_state, '.', color=colorcurr, label = 'State %d'%i_state)
        pl.plot(theta_probes, Ey_probes*fact_states[i_state], '-', color=colorcurr, label = 'Ref. %d'%i_state)
        pl.xlabel('theta [deg]')
        pl.ylabel('Ey [V/m]')
    #check single state case
    fact_single_state = 2.5
    single_state = pic.get_state_object()
    single_state.scatter(x_mp, y_mp, nel_mp*fact_single_state, charge = qe)
    pic.solve_states(single_state)
    Ex_prb_single_state, Ey_prb_single_state = single_state.gather(x_probes, y_probes)
    colorcurr = 'black'
    pl.subplot(2,1,1)
    pl.plot(theta_probes, Ex_prb_single_state, '.', color=colorcurr, label = 'Single state')
    pl.plot(theta_probes, Ex_probes*fact_single_state, '-', color=colorcurr, label = 'Single ref.')
    pl.subplot(2,1,2)
    pl.plot(theta_probes, Ey_prb_single_state, '.', color=colorcurr, label = 'Single state')
    pl.plot(theta_probes, Ey_probes*fact_single_state, '-', color=colorcurr, label = 'Single ref.')
    
    sp1.ticklabel_format(style='sci', scilimits=(0,0),axis='x') 
    sp1.ticklabel_format(style='sci', scilimits=(0,0),axis='y')
    sp1.legend(loc='center left', bbox_to_anchor=(1, 0.))
    sp2.ticklabel_format(style='sci', scilimits=(0,0),axis='x') 
    sp2.ticklabel_format(style='sci', scilimits=(0,0),axis='y')
    pl.subplots_adjust(left = .10, right = .77, bottom = .13,top = .90, hspace = .40)
    pl.suptitle(str(pic.__class__).replace('<','').replace('>','').split('.')[-1].replace("'", ''))

pl.show()
