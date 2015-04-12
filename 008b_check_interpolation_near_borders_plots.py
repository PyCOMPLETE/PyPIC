
Dh_mm=1.
import myloadmat_to_obj as mlo
import numpy as np
import pylab as pl
import mystyle as ms

linew = 2.
mksz = 10

label_plots = 'ellip_gaussian_'
ob_ref = mlo.myloadmat_to_obj('norepository_BE_Dh%.1fmm.mat'%Dh_mm)
ob_new = mlo.myloadmat_to_obj('norepository_FDSW_Dh%.1fmm.mat'%Dh_mm)
ob_old= mlo.myloadmat_to_obj('norepository_FDSW_Dh%.1fmm.mat'%Dh_mm)



N_points = len(ob_ref.xmax_test_list)
err_abs_list = []
err_abs_list_old = []
err_rel_list = []
err_rel_list_old = []

erry_rel_list_old = []
erry_rel_list = []

for ii in xrange(N_points): 

	Ex_ref = ob_ref.Ex[ii,:]
	Ey_ref = ob_ref.Ey[ii,:]
	
	Ex_new = ob_new.Ex[ii,:]
	Ey_new = ob_new.Ey[ii,:]
	
	Ex_old= ob_old.Ex[ii,:]
	Ey_old= ob_old.Ey[ii,:]

	err_abs =  np.sqrt(np.sum((Ex_ref-Ex_new)**2+(Ey_ref-Ey_new)**2))
	err_rel = err_abs/np.sqrt(np.sum((Ex_ref)**2+(Ey_ref)**2))
	
	err_abs_old =  np.sqrt(np.sum((Ex_ref-Ex_old)**2+(Ey_ref-Ey_old)**2))
	err_rel_old = err_abs_old/np.sqrt(np.sum((Ex_ref)**2+(Ey_ref)**2))
	
	erry_rel =  np.sqrt(np.sum((Ey_ref-Ey_new)**2))/np.sqrt(np.sum((Ey_ref)**2))
	erry_rel_old =  np.sqrt(np.sum((Ey_ref-Ey_old)**2))/np.sqrt(np.sum((Ey_ref)**2))
	
	err_abs_list.append(err_abs)
	err_rel_list.append(err_rel)
	
	err_abs_list_old.append(err_abs_old)
	err_rel_list_old.append(err_rel_old)
	
	erry_rel_list_old.append(erry_rel_old)
	erry_rel_list.append(erry_rel)
	
	
	
	#~ pl.figure(100)
	#~ pl.plot(Ex_ref, 'r')
	#~ pl.plot(Ex_new, 'b')
	#~ pl.plot(Ex_old, 'g')

na = np.array	
pl.close('all')	
ms.mystyle_arial(fontsz=16, dist_tick_lab=10)
pl.figure(1)	
pl.plot(1000*ob_ref.xmax_test_list, err_abs_list)
pl.xlim(0,1000*ob_ref.x_aper)

pl.figure(2)	
pl.plot(1000*(ob_ref.xmax_test_list), 100*na(err_rel_list))
pl.plot(1000*(ob_ref.xmax_test_list), 100*na(err_rel_list_old), 'r')
pl.xlim(0,1000*ob_ref.x_aper)

pl.figure(3)	
pl.plot(1000*(ob_ref.x_aper-ob_ref.xmax_test_list), 100*na(err_rel_list), '.-')
pl.plot(1000*(ob_ref.x_aper-ob_ref.xmax_test_list), 100*na(err_rel_list_old), '.-r')
pl.grid('on')
pl.xlim(0,None)

pl.figure(30)	
pl.plot(1000*(ob_ref.x_aper-ob_ref.xmax_test_list), 100*na(err_rel_list_old), '.-r', label = 'Old SC routine', linewidth = linew, markersize=mksz)
pl.plot(1000*(ob_ref.x_aper-ob_ref.xmax_test_list), 100*na(err_rel_list), '.-', label = 'New SC routine', linewidth = linew, markersize=mksz)
pl.grid('on')
pl.xlim(0,3)
pl.ylim(0,20)
pl.xlabel('Distance from edge [mm]')
pl.ylabel('Rms error [%]')
pl.legend(prop={'size':16})
fname = label_plots+'error_at_boudary_Dh%.1fmm'%Dh_mm
pl.suptitle(fname)
pl.savefig(fname+'.png', dpi=200)

#~ pl.figure(4)	
#~ pl.plot(1000*(ob_ref.x_aper-ob_ref.xmax_test_list), 100*na(erry_rel_list))
#~ pl.plot(1000*(ob_ref.x_aper-ob_ref.xmax_test_list), 100*na(erry_rel_list_old), 'r')
#~ pl.grid('on')
#~ pl.xlim(0,None)

x_obs = ob_ref.x_aper
i_obs = np.argmin(np.abs(ob_ref.xmax_test_list-x_obs))

pl.figure(100)
pl.clf()
sp1=pl.subplot(2,1,1)
pl.plot(180./np.pi*ob_ref.theta, ob_new.Ex[i_obs,:], '.-r', linewidth = linew, markersize=mksz)
pl.plot(180./np.pi*ob_ref.theta,ob_ref.Ex[i_obs,:], 'g', linewidth = linew)
pl.ylabel('Ex [V/m]')
ms.sciy()
pl.subplot(2,1,2, sharex=sp1)
pl.plot(180./np.pi*ob_ref.theta,ob_new.Ey[i_obs,:], '.-r', linewidth = linew, markersize=mksz)
pl.plot(180./np.pi*ob_ref.theta,ob_ref.Ey[i_obs,:], 'g', linewidth = linew)
pl.xlim(0,360)
ms.sciy()
pl.ylabel('Ey [V/m]')
pl.xlabel('Theta [deg]')
pl.subplots_adjust(hspace=.3)
fname = label_plots+'new_sc'+'_field_at_boudary_Dh%.1fmm'%Dh_mm
pl.suptitle(fname)
pl.savefig(fname+'.png', dpi=200)

pl.figure(101)
pl.clf()
sp1=pl.subplot(2,1,1)
pl.plot(180./np.pi*ob_ref.theta, ob_old.Ex[i_obs,:], '.-r', linewidth = linew, markersize=mksz)
pl.plot(180./np.pi*ob_ref.theta,ob_ref.Ex[i_obs,:], 'g', linewidth = linew)
pl.ylabel('Ex [V/m]')
ms.sciy()
pl.subplot(2,1,2, sharex=sp1)
pl.plot(180./np.pi*ob_ref.theta,ob_old.Ey[i_obs,:], '.-r', linewidth = linew, markersize=mksz)
pl.plot(180./np.pi*ob_ref.theta,ob_ref.Ey[i_obs,:], 'g', linewidth = linew)
pl.xlim(0,360)
ms.sciy()
pl.ylabel('Ey [V/m]')
pl.xlabel('Theta [deg]')
pl.subplots_adjust(hspace=.3)
fname = label_plots+'old_sc'+'_field_at_boudary_Dh%.1fmm'%Dh_mm
pl.suptitle(fname)
pl.savefig(fname+'.png', dpi=200)


dist_from_bou_list = [1e-3, 2e-3, 5e-3, 10e-3]

for dist_from_bou in dist_from_bou_list:
	x_obs = ob_ref.x_aper-dist_from_bou
	i_obs = np.argmin(np.abs(ob_ref.xmax_test_list-x_obs))

	pl.figure(200)
	pl.clf()
	sp1=pl.subplot(2,1,1)
	pl.plot(180./np.pi*ob_ref.theta, ob_new.Ex[i_obs,:], '.-r', linewidth = linew, markersize=mksz)
	pl.plot(180./np.pi*ob_ref.theta,ob_ref.Ex[i_obs,:], 'g', linewidth = linew)
	pl.ylabel('Ex [V/m]')
	ms.sciy()
	pl.subplot(2,1,2, sharex=sp1)
	pl.plot(180./np.pi*ob_ref.theta,ob_new.Ey[i_obs,:], '.-r', linewidth = linew, markersize=mksz)
	pl.plot(180./np.pi*ob_ref.theta,ob_ref.Ey[i_obs,:], 'g', linewidth = linew)
	pl.xlim(0,360)
	ms.sciy()
	pl.ylabel('Ey [V/m]')
	pl.xlabel('Theta [deg]')
	pl.subplots_adjust(hspace=.3)
	fname = label_plots+'new_sc'+'_field_at_%.1fmm_from_boudary_Dh%.1fmm'%(dist_from_bou/1e-3, Dh_mm)
	pl.suptitle(fname)
	pl.savefig(fname+'.png', dpi=200)

	pl.figure(201)
	pl.clf()
	sp1=pl.subplot(2,1,1)
	pl.plot(180./np.pi*ob_ref.theta, ob_old.Ex[i_obs,:], '.-r', linewidth = linew, markersize=mksz)
	pl.plot(180./np.pi*ob_ref.theta,ob_ref.Ex[i_obs,:], 'g', linewidth = linew)
	pl.ylabel('Ex [V/m]')
	ms.sciy()
	pl.subplot(2,1,2, sharex=sp1)
	pl.plot(180./np.pi*ob_ref.theta,ob_old.Ey[i_obs,:], '.-r', linewidth = linew, markersize=mksz)
	pl.plot(180./np.pi*ob_ref.theta,ob_ref.Ey[i_obs,:], 'g', linewidth = linew)
	pl.xlim(0,360)
	ms.sciy()
	pl.ylabel('Ey [V/m]')
	pl.xlabel('Theta [deg]')
	pl.subplots_adjust(hspace=.3)
	fname = label_plots+'old_sc'+'_field_at_%.1fmm_from_boudary_Dh%.1fmm'%(dist_from_bou/1e-3, Dh_mm)
	pl.suptitle(fname)
	pl.savefig(fname+'.png', dpi=200)

pl.show()
    
