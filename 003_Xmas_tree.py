import pylab as pl
import numpy as np
from scipy import rand
from . import geom_impact_poly as poly
from . import FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
from . import FFT_OpenBoundary_SquareGrid as PIC_FFT
from . import FFT_PEC_Boundary_SquareGrid as PIC_PEC_FFT
try:
    from CyFPPS import PyFPPS as PIC_FPPS
except ImportError:
    print("Not possible to import PyFPPS, replaced with FFT_Open")
    PIC_FPPS = None

na = np.array
Dh =1e-1
N_part_gen = 100000

tree = [[0,0],
		[1.,0],
		[1., 1,],
		[5.,1.],
		[2.,4.],
		[4,4],
		[2,7],
		[3,7],
		[1,9],
		[2,9],
		[0,11]]
		
tree=np.array(tree)
x_tree = tree[:,0]
y_tree = tree[:,1]

y_tree -= 6.

x_aper = 6.
y_aper = 7.

x_tree = np.array([0.]+ list(x_tree)+[0.])
y_tree = np.array([-y_aper]+ list(y_tree)+[y_aper])


		


x_part = x_aper*(2.*rand(N_part_gen)-1.)
y_part = y_aper*(2.*rand(N_part_gen)-1.)

x_on_tree = np.interp(y_part, y_tree, x_tree)

mask_keep = np.logical_and(np.abs(x_part)<x_on_tree, np.abs(x_part)>x_on_tree*0.8)
x_part = x_part[mask_keep]
y_part = y_part[mask_keep]

nel_part = 0*x_part+1.


		


chamber = poly.polyg_cham_geom_object({'Vx':na([x_aper, -x_aper, -x_aper, x_aper]),
									   'Vy':na([y_aper, y_aper, -y_aper, -y_aper]),
									   'x_sem_ellip_insc':0.99*x_aper,
									   'y_sem_ellip_insc':0.99*y_aper})
									   
picFDSW = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb = chamber, Dh = Dh)
picFFTPEC = PIC_PEC_FFT.FFT_PEC_Boundary_SquareGrid(x_aper = chamber.x_aper, y_aper = chamber.y_aper, Dh = Dh)
picFFT = PIC_FFT.FFT_OpenBoundary_SquareGrid(x_aper = chamber.x_aper, y_aper = chamber.y_aper, Dh = Dh)
R_cham = np.sqrt((chamber.x_aper/2)**2 + (chamber.y_aper/2)**2 )
if PIC_FPPS: picFPPS = PIC_FPPS(nTheta=100,nR=200,a=R_cham,solverType='OpenBoundary')

picFDSW.scatter(x_part, y_part, nel_part)
picFFTPEC.scatter(x_part, y_part, nel_part)
picFFT.scatter(x_part, y_part, nel_part)
if PIC_FPPS: picFPPS.scatter(x_part,y_part,nel_part)


picFDSW.solve()
picFFTPEC.solve()
picFFT.solve()
if PIC_FPPS: picFPPS.solve()

pl.close('all')
pl.figure(1)
#pl.plot(x_tree, y_tree, '-o')
pl.plot(x_part, y_part, '.g', markersize=2)
pl.axis('equal')
pl.suptitle('Macroparticle positions')
pl.savefig('Xmas_MPs.png', dpi=200)

pl.figure(2)
pl.pcolor(picFFTPEC.rho.T)
pl.axis('equal')
pl.suptitle('Charge density')
pl.savefig('Xmas_rho.png', dpi=200)

pl.figure(3)
pl.pcolor((picFFTPEC.efx**2+picFFTPEC.efy**2).T)
pl.axis('equal')
pl.suptitle('Magnitude electric field\nFFT')
pl.colorbar()
pl.savefig('Xmas_efield_FFT.png', dpi=200)

pl.figure(4)
pl.pcolor(picFFTPEC.phi.T)
pl.colorbar()
pl.axis('equal')


pl.figure(102)
pl.pcolor(picFDSW.rho.T)
pl.axis('equal')
pl.suptitle('Charge density')


pl.figure(103)
pl.pcolor((picFDSW.efx**2+picFDSW.efy**2).T)
pl.axis('equal')
pl.suptitle('Magnitude electric field\nFinite differences')
pl.colorbar()
pl.savefig('Xmas_efield_FD.png', dpi=200)

pl.figure(104)
pl.pcolor(picFDSW.phi.T)
pl.colorbar()
pl.axis('equal')

pl.figure(203)
pl.pcolor((picFFT.efx**2+picFFT.efy**2).T)
pl.axis('equal')
pl.suptitle('Magnitude electric field - free space')
pl.colorbar()
pl.savefig('Xmas_efield_open_boudary.png', dpi=200)

if PIC_FPPS: 
    x = np.arange(-chamber.x_aper,chamber.x_aper,2*chamber.x_aper/300)
    y = np.arange(-chamber.y_aper,chamber.y_aper,2*chamber.y_aper/300)
    X,Y = np.meshgrid(x,y)
    E = np.zeros_like(X)
    for i in range(np.shape(X)[0]):
        Ex_FPPS,Ey_FPPS = picFPPS.gather(X[i,:],Y[i,:])
        print((np.shape(Ex_FPPS)))
        E[i,:] = Ex_FPPS**2+Ey_FPPS**2
    pl.figure(303)
    pl.pcolor(X,Y,E)
    pl.axis('equal')
    pl.suptitle('Magnitude electric field - free space')
    pl.colorbar()
    pl.savefig('Xmas_efield_FPPS_openBoundary.png', dpi=200)


pl.show()
