from __future__ import division

from pylab import *
import numpy as np
import scipy as sp
import pylab as pl



def fast_dst(x):
# Perform a fast DST on a vector. Since Matlab does not have a DST function,
# this just uses the built in FFT function.
#
# parameter:
# x = input column vector #
# returns:
# y=theDSTofx
	n = len(x);
	tmp = zeros((2*n + 2));
	#tmp = -imag(fft([0; x; zeros((n+1,1))]));
	tmp[1:n+1]=x
	tmp=-(sp.fft(tmp).imag)
	y = tmp[1:n+1];
	return y

def fft_poisson(b,h):
# This function solves the 2-d Poisson problem del^2 u = f(x,y) using the fast
# Fourier transformation. The Poisson problem has homogeneous Dirichlet boundary
# conditions, and is defined on a square region with equal sized mesh widths.
#
# parameters:
# b=matrixoff values evaluated at interior meshpoints
#   h = mesh width
#
# returns:
#   u = solution to PDE at interior meshpoints
# get dimensions of b
	m, n = b.shape;
	b_bar = np.zeros((n,n));
	u_bar = b_bar;
	u = u_bar;
	# make sure we have a square grid
	if (m != n):
	    raise ValueError('matrix b is not square');
	
	# b_bar = 2/(n+1) * v * b * v
	# first do a DST on columns of b, which is the same as multiplying v*b
	for j in xrange(n):
	    b_bar[:,j] = fast_dst(b[:,j]);
	
	# then do DST on rows of vb, which is analogous to multiplying v*b*v
	for i in xrange(n):
	    b_bar[i,:] = fast_dst(b_bar[i,:]);
	    # have to take transpose of row, since fast_dst needs a column vector
	    #b_bar[i,:] = fast_dst(b_bar[i,:].T).T;

	# now scale by 2/(n+1)
	b_bar = b_bar * (2/n+1);
	# next we can solve for u_bar
	u_bar = np.zeros((n,n));
	lam = arange(1,n+0.5,1);
	lam = -4 * (sin((lam*np.pi) / (2*n + 2)))**2;
	for i in xrange(n):
	    for j in xrange(n):
	        u_bar[i,j] = (h**2 * b_bar[i,j]) / (lam[i] + lam[j]);
	
	# u = 2/(n+1) * v * u_bar * v
	# do a DST on columns of u_bar, which is analogous to multiplying v * u_bar
	for j in xrange(n):
	    u[:,j] = fast_dst(u_bar[:,j]);

	# then do a DST on rows
	for i in xrange(n):
	    u[i,:] = fast_dst(u[i,:]);

	# last, multiply by 2/(n+1)
	u = u * 2/(n+1);
	return u


# solve del^2 u = -4 * pi * rho using a fast Poisson solver
# domain: for simplicity, a square {0 <= x,y <= 1}. the edges are grounded, so
# on the boundary u(x,y) = 0. we'll have 100 cells in the mesh of our domain.
a = 0; b = 1;
ncells = 100;
xe = linspace(a,b,ncells+1); ye = linspace(a,b,ncells+1);
x, y = meshgrid(xe,ye);

# grid spacing
h = (b-a)/ncells;
# set up right hand side. our charge density will just be 10 point charges on
# random grid points with strength 1. note that the right hand side does not
# include the borders; the charge is always zero on the borders since they are
# have no potential (they are grounded).
rho = zeros((ncells-1,ncells-1));
pts = np.floor(sp.rand(10,2) * (ncells-1));
for i in xrange(10):
    rho[pts[i,0],pts[i,1]] = -4 * np.pi;

# now solve to get a potential field
V = fft_poisson(rho,h);


pl.close('all')
pl.figure(1)
pl.pcolor(V)
pl.colorbar()
pl.show()

# # pad V with zero boundary values
# V = [zeros(ncells-1,1), V, zeros(ncells-1,1)];
# V = [zeros(1,ncells+1); V; zeros(1,ncells+1)];
# # contour plot of potential field
# figure(1)
# contour(x,y,V)
# xlabel('x'); ylabel('y');
# # mesh plot of potential field
# figure(2)
# mesh(x,y,V)
# xlabel('x'); ylabel('y');

