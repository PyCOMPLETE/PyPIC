from __future__ import division

from pylab import *
import numpy as np
import scipy as sp
import pylab as pl



def fast_dst(x):
# Perform a fast DST on a vector. Since Matlab does not have a DST function,
# this just uses the built in FFT function.

	n = len(x);
	tmp = zeros((2*n + 2));
	tmp[1:n+1]=x
	tmp=-(sp.fft(tmp).imag)
	y = sqrt(2/(n+1))*tmp[1:n+1];
	return y

def fft_poisson(b,h):

	
	m, n = b.shape;
	b_bar = np.zeros((m,n));
	u_bar = b_bar;
	u = u_bar;

	if (m != n):
	    print 'Warning: matrix b is not square'

	
	for j in xrange(n):
	    b_bar[:,j] = fast_dst(b[:,j]);
	

	for i in xrange(m):
	    b_bar[i,:] = fast_dst(b_bar[i,:]);

	xx = arange(1,m+0.5,1);
	yy = arange(1,n+0.5,1);
	
	YY, XX = np.meshgrid(yy,xx) 
	green = -4*(sin(XX/2*pi/(m+1))**2/h**2+\
               sin(YY/2*pi/(n+1))**2/h**2);
               
	u_bar = b_bar/green    
	

	for j in xrange(n):
	    u[:,j] = fast_dst(u_bar[:,j]);

	for i in xrange(m):
	    u[i,:] = fast_dst(u[i,:]);


	return u




