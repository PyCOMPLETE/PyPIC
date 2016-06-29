
import numpy as np
import pylab as pl
from scipy.constants import e as qe

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
N_probes = 1000.
n_sigma_probes = 1.
theta_probes = np.linspace(0, 2*np.pi, N_probes)
x_probes = n_sigma_probes*sigmax*np.cos(theta_probes)
y_probes = n_sigma_probes*sigmay*np.sin(theta_probes)

# build pic
Dh=1e-3
import FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
pic = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh)


pic.scatter(x_mp, y_mp, nel_mp, charge = qe)

pic.solve()

Ex_probes, Ey_probes = pic.gather(x_probes, y_probes)


pl.close('all')
pl.figure(1)
pl.subplot(2,1,1)
pl.plot(theta_probes, Ex_probes)
pl.subplot(2,1,2)
pl.plot(theta_probes, Ey_probes)

fact_states = np.linspace(1.1, 1.5, 4)
N_states = len(fact_states)

list_states = []
for _ in fact_states:
    list_states.append(pic.get_state_object())
    
for i_state, state in enumerate(list_states):
    state.scatter(x_mp, y_mp, nel_mp*fact_states[i_state], charge = qe)
    
pic.

pl.show()
