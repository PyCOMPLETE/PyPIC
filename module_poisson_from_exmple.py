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
	b_bar = np.zeros((m,n));
	u_bar = b_bar;
	u = u_bar;
	# make sure we have a square grid
	if (m != n):
	    print 'Warning: matrix b is not square'
	    #raise ValueError('matrix b is not square');
	
	# b_bar = 2/(n+1) * v * b * v
	# first do a DST on columns of b, which is the same as multiplying v*b
	for j in xrange(n):
	    b_bar[:,j] = fast_dst(b[:,j]);
	
	# then do DST on rows of vb, which is analogous to multiplying v*b*v
	for i in xrange(m):
	    b_bar[i,:] = fast_dst(b_bar[i,:]);
	    # have to take transpose of row, since fast_dst needs a column vector
	    #b_bar[i,:] = fast_dst(b_bar[i,:].T).T;

	# now scale by 2/(n+1)
	#b_bar = b_bar * (2/n+1);
	# next we can solve for u_bar
	u_bar = np.zeros((m,n));
	#~ lam = arange(1,n+0.5,1);
	#~ lam = -4 * (sin((lam*np.pi) / (2*n + 2)))**2;
	#~ for i in xrange(n):
	    #~ for j in xrange(n):
	        #~ u_bar[i,j] = (h**2 * b_bar[i,j]) / (lam[i] + lam[j]);
	xx = arange(1,m+0.5,1);
	yy = arange(1,n+0.5,1);
	
	YY, XX = np.meshgrid(yy,xx) 
	green = -4*(sin(XX/2*pi/(m+1))**2/h**2+\
               sin(YY/2*pi/(n+1))**2/h**2);
               
	u_bar = b_bar/green    
	
	
	# u = 2/(n+1) * v * u_bar * v
	# do a DST on columns of u_bar, which is analogous to multiplying v * u_bar
	for j in xrange(n):
	    u[:,j] = fast_dst(u_bar[:,j]);

	# then do a DST on rows
	for i in xrange(m):
	    u[i,:] = fast_dst(u[i,:]);

	# last, multiply by 2/(n+1)
	#u = u * 2/(n+1);
	return u




