import sys, os
BIN=os.path.expanduser('../')
sys.path.append(BIN)
BIN=os.path.expanduser('../PyHEADTAIL/')
sys.path.append(BIN)
BIN=os.path.expanduser('../PyHEADTAIL/PyHEADTAIL/testing/script-tests/')
sys.path.append(BIN)
from LHC import LHC
import PyPIC.geom_impact_ellip as ell
import PyPIC.FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
import PyPIC.Bassetti_Erskine as PIC_BE
from PyPIC.MultiGrid import AddTelescopicGrids
from scipy.constants import e, epsilon_0,c
import numpy as np
import pylab as pl
import mystyle as ms

qe = e
eps0 = epsilon_0

p0_GeV = 2000.

# LHC
machine_configuration='6.5_TeV_collision_tunes'
intensity=1.2e11
epsn_x=2.5e-6
epsn_y=2.5e-6
sigma_z=7e-2

n_macroparticles=1000000

sparse_solver = 'PyKLU'

machine = LHC(machine_configuration = machine_configuration, optics_mode = 'smooth', n_segments = 1, p0=p0_GeV*1e9*e/c)


# generate beam
bunch = machine.generate_6D_Gaussian_bunch(n_macroparticles = n_macroparticles, intensity = intensity, 
                            epsn_x = epsn_x, epsn_y = epsn_y, sigma_z = sigma_z)


# Single grid parameters
Dh_single = 0.5*bunch.sigma_x() #.3

# Bassetti-Erskine parameters
Dh_BE = 0.2*bunch.sigma_x()

#  Multi grid parameters
Dh_single_ext = 1e-3
Sx_target = 10*bunch.sigma_x()
Sy_target = 10*bunch.sigma_y()
Dh_target = 0.5*bunch.sigma_x()#.3

# chamber parameters
x_aper = 22e-3
y_aper = 18e-3

# build chamber
chamber = ell.ellip_cham_geom_object(x_aper = x_aper, y_aper = y_aper)
Vx, Vy = chamber.points_on_boundary(N_points=200)

# build Bassetti Erskine 
pic_BE = PIC_BE.Interpolated_Bassetti_Erskine(x_aper=x_aper, y_aper=y_aper, Dh=Dh_BE, sigmax=bunch.sigma_x(), sigmay=bunch.sigma_y(), 
		n_imag_ellip=20, tot_charge=bunch.intensity*bunch.charge)
		
# build single grid pic
pic_singlegrid = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh_single, sparse_solver = sparse_solver)

# build single grid pic for telescope
pic_singlegrid_ext = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh_single_ext, sparse_solver = sparse_solver)

# build telescope
pic_multigrid = AddTelescopicGrids(pic_main = pic_singlegrid_ext, f_telescope = 0.3, 
    target_grid = {'x_min_target':-Sx_target/2., 'x_max_target':Sx_target/2.,'y_min_target':-Sy_target/2.,'y_max_target':Sy_target/2.,'Dh_target':Dh_target}, 
    N_nodes_discard = 3., N_min_Dh_main = 10, sparse_solver=sparse_solver)

pic_singlegrid.scatter(bunch.x, bunch.y, bunch.particlenumber_per_mp+bunch.y*0., charge=qe)
pic_multigrid.scatter(bunch.x, bunch.y, bunch.particlenumber_per_mp+bunch.y*0., charge=qe)          
                                                            
#scatter and solve     
#pic solve timing
import time
N_rep_test_single = 1000
print 'Solving PIC single %d times'%N_rep_test_single
t_start = time.mktime(time.localtime())
for _ in xrange(N_rep_test_single):
    pic_singlegrid.solve()
t_stop = time.mktime(time.localtime())
t_sw_single = (t_stop-t_start)/N_rep_test_single   
print 'solving time singlegrid ', t_sw_single

N_rep_test_multi = 10000                                              
print 'Solving PIC multi %d times'%N_rep_test_multi
t_start = time.mktime(time.localtime())
for _ in xrange(N_rep_test_multi):
    pic_multigrid.solve()
t_stop = time.mktime(time.localtime())
t_sw_multi = (t_stop-t_start)/N_rep_test_multi
print 'solving time multigrid ', t_sw_multi


# build probes for single circle
theta=np.linspace(0., 2*np.pi, 1000)
n_sigma_probes = 1.
x_probes = n_sigma_probes*bunch.sigma_x()*np.cos(theta)
y_probes = n_sigma_probes*bunch.sigma_y()*np.sin(theta)  

# get field at probes
Ex_BE, Ey_BE = pic_BE.gather(x_probes, y_probes)
Ex_singlegrid, Ey_singlegrid = pic_singlegrid.gather(x_probes, y_probes)
Ex_multigrid, Ey_multigrid = pic_multigrid.gather(x_probes, y_probes)


#plots
pl.close('all')
ms.mystyle_arial(fontsz=14)

#electric field at probes


pl.figure(1, figsize=(18,6)).patch.set_facecolor('w')
pl.subplot(1,3,1)
#~ pl.plot(pic_singlegrid.xn, pic_singlegrid.yn,'.y', label = 'Singlegrid')
#~ pl.plot(pic_singlegrid_ext.xn, pic_singlegrid_ext.yn,'.m', label = 'Singlegrid telescope')
for ii in xrange(pic_multigrid.n_grids):
    pl.plot(pic_multigrid.pic_list[ii].pic_internal.chamb.Vx, pic_multigrid.pic_list[ii].pic_internal.chamb.Vy, '.-', label = 'Internal grid %d'%ii)
pl.plot(bunch.x, bunch.y, '.k')
pl.plot(Vx, Vy, 'k--', label = 'Chamber')
pl.plot(x_probes, y_probes, 'c--', label = 'probe')
pl.xlabel('x [m]')
pl.ylabel('y [m]')
pl.ticklabel_format(style='sci', scilimits=(0,0),axis='x') 
pl.ticklabel_format(style='sci', scilimits=(0,0),axis='y')
pl.axis('equal')
pl.legend(loc='best')
pl.subplot(1,3,2)
pl.plot(theta*180/np.pi, Ex_BE, 'k--', label = 'BE')
pl.plot(theta*180/np.pi, Ex_singlegrid, '.-g', label = 'Singlegrid')
pl.plot(theta*180/np.pi, Ex_multigrid, '.-r', label = 'Multigrid')
pl.xlabel('theta[deg]')
pl.ylabel('Ex [V/m] ')
pl.ticklabel_format(style='sci', scilimits=(0,0),axis='x') 
pl.ticklabel_format(style='sci', scilimits=(0,0),axis='y')
pl.grid()
pl.legend(loc='best')
pl.subplot(1,3,3)
pl.plot(theta*180/np.pi, Ey_BE, 'k--', label = 'BE')
pl.plot(theta*180/np.pi, Ey_singlegrid, '.-g', label = 'Singlegrid')
pl.plot(theta*180/np.pi, Ey_multigrid, '.-r', label = 'Multigrid')
pl.xlabel('theta[deg]')
pl.ylabel('Ey [V/m] ')
pl.ticklabel_format(style='sci', scilimits=(0,0),axis='x') 
pl.ticklabel_format(style='sci', scilimits=(0,0),axis='y')
pl.grid()
pl.legend(loc='best')
pl.suptitle('Probe @ %.1f sigmans'%n_sigma_probes)
pl.tight_layout()


# plot RMS error vs distance 
r_probes_val = []
r_min = 1e-4
r_max = 15e-3
RMSE_singlegrid = []
RMSE_multigrid = []
for r_probes in np.logspace(np.log10(r_min), np.log10(r_max), 100):
	theta=np.linspace(0., 2*np.pi, 100)
	x_probes = r_probes*np.cos(theta)
	y_probes = r_probes*np.sin(theta)
	r_probes_val.append(r_probes)
	# pic gather
	Ex_BE, Ey_BE = pic_BE.gather(x_probes, y_probes)
	Ex_singlegrid, Ey_singlegrid = pic_singlegrid.gather(x_probes, y_probes)
	Ex_multigrid, Ey_multigrid = pic_multigrid.gather(x_probes, y_probes)
	# RMS	
	RMSE_singlegrid.append(np.sqrt(np.sum((Ex_singlegrid-Ex_BE)**2+(Ey_singlegrid-Ey_BE)**2))/np.sqrt(np.sum((Ex_BE)**2+(Ey_BE)**2)))
	RMSE_multigrid.append(np.sqrt(np.sum((Ex_multigrid-Ex_BE)**2+(Ey_multigrid-Ey_BE)**2))/np.sqrt(np.sum((Ex_BE)**2+(Ey_BE)**2)))
pl.figure(2).patch.set_facecolor('w')
pl.loglog(r_probes_val, RMSE_singlegrid, '.-r', label = 'Single grid (t=%.1f ms)'%(t_sw_single*1000.), linewidth=2, markersize=10)
pl.loglog(r_probes_val, RMSE_multigrid, '.-b', label = 'Multi grid (t=%.1f ms)'%(t_sw_multi*1000.), linewidth=2, markersize=10)
pl.xlim(r_min, r_max)
pl.xlabel('r [m]')
pl.ylabel('RMS error')
pl.suptitle('$\sigma_x$ = %.2e m $\sigma_y$ = %.2e m\n $\Delta h_{single}$ = %.2e m $\Delta h_{multi}$ = %.2e m\n $\Delta h_{BE}$ = %.2e m\n Solving time: $t_{single}$ = %.1f ms, $t_{multi}$ = %.1f ms'%(bunch.sigma_x(), bunch.sigma_y(), Dh_single, 
                Dh_target, Dh_BE, t_sw_single*1000., t_sw_multi*1000.))
pl.subplots_adjust(bottom = .13, top = .75)
pl.grid()
pl.legend(loc='best', prop={'size':14})


# plot RMS error vs sigma
n_probes = 100
n_sigma_min = 0.1
n_sigma_max = 100
n_sigma_probes = np.logspace(np.log10(n_sigma_min), np.log10(n_sigma_max), n_probes)
RMSE_singlegrid = []
RMSE_multigrid = []
for n_sigma_probe in n_sigma_probes:
	theta=np.linspace(0., 2*np.pi, 100)
	x_probes = n_sigma_probe*bunch.sigma_x()*np.cos(theta)
	y_probes = n_sigma_probe*bunch.sigma_y()*np.sin(theta)
	# pic gather
	Ex_BE, Ey_BE = pic_BE.gather(x_probes, y_probes)
	Ex_singlegrid, Ey_singlegrid = pic_singlegrid.gather(x_probes, y_probes)
	Ex_multigrid, Ey_multigrid = pic_multigrid.gather(x_probes, y_probes)
	# RMS	
	RMSE_singlegrid.append(np.sqrt(np.sum((Ex_singlegrid-Ex_BE)**2+(Ey_singlegrid-Ey_BE)**2))/np.sqrt(np.sum((Ex_BE)**2+(Ey_BE)**2)))
	RMSE_multigrid.append(np.sqrt(np.sum((Ex_multigrid-Ex_BE)**2+(Ey_multigrid-Ey_BE)**2))/np.sqrt(np.sum((Ex_BE)**2+(Ey_BE)**2)))
pl.figure(3).patch.set_facecolor('w')
pl.loglog(n_sigma_probes, RMSE_singlegrid, '.-r', label = 'Single grid (t=%.1f ms)'%(t_sw_single*1000.), linewidth=3)
pl.loglog(n_sigma_probes, RMSE_multigrid, '.-b', label = 'Multi grid (t=%.1f ms)'%(t_sw_multi*1000.), linewidth=3)
pl.xlabel('$\sigma$')
pl.ylabel('RMS error')
pl.title('$\sigma_x$ = %.2e m $\sigma_y$ = %.2e m  \n $\Delta h_{single}$ = %.2e m $\Delta h_{multi}$ = %.2e [m]\n $\Delta h_{BE}$ = %.2e [m]\n Solving time: $t_{single}$ = %.1f ms, $t_{multi}$ = %.1f ms'%(bunch.sigma_x(), bunch.sigma_y(), Dh_single, 
                Dh_target, Dh_BE, t_sw_single*1000., t_sw_multi*1000.))
pl.subplots_adjust(bottom = .13, top = .70)
pl.grid()
pl.legend(loc='best')


#plot error map
Dh_test = Dh_target
x_grid_probes = np.arange(np.min(pic_singlegrid.xg), np.max(pic_singlegrid.xg)+Dh_test, Dh_test)
y_grid_probes = np.arange(np.min(pic_singlegrid.yg), np.max(pic_singlegrid.yg), Dh_test)
[xn, yn]=np.meshgrid(x_grid_probes,y_grid_probes)
xn=xn.T
xn=xn.flatten()
yn=yn.T
yn=yn.flatten()
#pic gather
Ex_BE_n, Ey_BE_n = pic_BE.gather(xn, yn)	
Ex_BE_matrix=np.reshape(Ex_BE_n,(len(y_grid_probes),len(x_grid_probes)), 'F').T
Ey_BE_matrix=np.reshape(Ey_BE_n,(len(y_grid_probes),len(x_grid_probes)), 'F').T

Ex_singlegrid_n, Ey_singlegrid_n = pic_singlegrid.gather(xn, yn)	
Ex_singlegrid_matrix=np.reshape(Ex_singlegrid_n,(len(y_grid_probes),len(x_grid_probes)), 'F').T
Ey_singlegrid_matrix=np.reshape(Ey_singlegrid_n,(len(y_grid_probes),len(x_grid_probes)), 'F').T

Ex_multigrid_n, Ey_multigrid_n = pic_multigrid.gather(xn, yn)	
Ex_multigrid_matrix=np.reshape(Ex_multigrid_n,(len(y_grid_probes),len(x_grid_probes)), 'F').T
Ey_multigrid_matrix=np.reshape(Ey_multigrid_n,(len(y_grid_probes),len(x_grid_probes)), 'F').T

pl.figure(4, figsize=(12, 6)).patch.set_facecolor('w')
sp1 = pl.subplot(121)
pl.pcolormesh(x_grid_probes, y_grid_probes, 
	np.log10(np.sqrt((((Ex_singlegrid_matrix-Ex_BE_matrix)**2+(Ey_singlegrid_matrix-Ey_BE_matrix)**2)/(Ex_BE_matrix**2+Ey_BE_matrix**2)))).T,
	vmax=0., vmin=-7.0)
pl.title('RMS error Singlegrid - BE')
pl.xlabel('x [m]')
pl.ylabel('y [m]')
cb=pl.colorbar(); pl.axis('equal')
cb.formatter.set_powerlimits((0, 0))
cb.update_ticks()
cb.set_label('RMS error')
sp1.ticklabel_format(style='sci', scilimits=(0,0),axis='x') 
sp1.ticklabel_format(style='sci', scilimits=(0,0),axis='y')

sp2 = pl.subplot(122, sharex = sp1, sharey = sp1)
pl.pcolormesh(x_grid_probes, y_grid_probes, 
	np.log10(np.sqrt((((Ex_multigrid_matrix-Ex_BE_matrix)**2+(Ey_multigrid_matrix-Ey_BE_matrix)**2)/(Ex_BE_matrix**2+Ey_BE_matrix**2)))).T,
	vmax=0., vmin=-7.0)
pl.title('RMS error Multigrid - BE')
pl.xlabel('x [m]')
pl.ylabel('y [m]')
cb=pl.colorbar(); pl.axis('equal')
cb.formatter.set_powerlimits((0, 0))
cb.update_ticks()
cb.set_label('RMS error')
sp1.ticklabel_format(style='sci', scilimits=(0,0),axis='x') 
sp1.ticklabel_format(style='sci', scilimits=(0,0),axis='y')
pl.subplots_adjust(bottom = .13,top = .70)
pl.suptitle('$\sigma_x$ = %.2e [m]\n $\sigma_y$ = %.2e [m]  \n $\Delta h_{single}$ = %.2e [m]\n $\Delta h_{multi}$ = %.2e [m]\n $\Delta h_{BE}$ = %.2e [m]\n Solving time: $t_{single}$ = %.1f ms, $t_{multi}$ = %.1f ms'%(bunch.sigma_x(), bunch.sigma_y(), Dh_single, 
                Dh_target, Dh_BE, t_sw_single*1000., t_sw_multi*1000.))

pl.show()
