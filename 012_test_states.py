
import numpy as np
import pylab as pl
from scipy.constants import e as qe
import mystyle as ms

# build chamber
x_aper = 2e-2; y_aper=1e-2
import geom_impact_ellip as ell
chamber = ell.ellip_cham_geom_object(x_aper = x_aper, y_aper = y_aper)

#build particle distribution
from scipy import randn
N_part = 10000; sigmax=1e-3; sigmay=2e-3
x_mp = sigmax*randn(N_part);
y_mp = sigmay*randn(N_part);
nel_mp = x_mp*0.+1.

#build probes
N_probes = 100
n_sigma_probes = 1.
theta_probes = np.linspace(0, 2*np.pi, N_probes)
x_probes = n_sigma_probes*sigmax*np.cos(theta_probes)
y_probes = n_sigma_probes*sigmay*np.sin(theta_probes)

# build pic
Dh=1e-3
import FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
pic = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh)

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
pl.close('all')
pl.figure(1)
pl.subplot(2,1,1)
pl.plot(theta_probes, Ex_probes)
pl.subplot(2,1,2)
pl.plot(theta_probes, Ey_probes)

for i_state, state in enumerate(list_states):
    colorcurr = ms.colorprog(i_state, N_states)
    Ex_prb_state, Ey_prb_state = state.gather(x_probes, y_probes)
    pl.subplot(2,1,1)
    pl.plot(theta_probes, Ex_prb_state, '.', color=colorcurr)
    pl.plot(theta_probes, Ex_probes*fact_states[i_state], '-', color=colorcurr)
    pl.subplot(2,1,2)
    pl.plot(theta_probes, Ey_prb_state, '.', color=colorcurr)
    pl.plot(theta_probes, Ey_probes*fact_states[i_state], '-', color=colorcurr)

#check single state case
fact_single_state = 2.5
single_state = pic.get_state_object()
single_state.scatter(x_mp, y_mp, nel_mp*fact_single_state, charge = qe)
pic.solve_states(single_state)
Ex_prb_single_state, Ey_prb_single_state = state.gather(x_probes, y_probes)
colorcurr = 'black'
pl.subplot(2,1,1)
pl.plot(theta_probes, Ex_prb_state, '.', color=colorcurr)
pl.plot(theta_probes, Ex_probes*fact_single_state, '-', color=colorcurr)
pl.subplot(2,1,2)
pl.plot(theta_probes, Ey_prb_state, '.', color=colorcurr)
pl.plot(theta_probes, Ey_probes*fact_single_state, '-', color=colorcurr)

pl.show()
