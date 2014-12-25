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
	
def dst2(x):
	m, n = x.shape;
	x_bar = np.zeros((m,n)); 
	
	for j in xrange(n):
	    x_bar[:,j] = fast_dst(x[:,j]);
	
	for i in xrange(m):
	    x_bar[i,:] = fast_dst(x_bar[i,:])
	    
	return x_bar

def fft_poisson(b,h):

	
	m, n = b.shape;

	xx = arange(1,m+0.5,1);
	yy = arange(1,n+0.5,1);
	
	YY, XX = np.meshgrid(yy,xx) 
	green = -4*(sin(XX/2*pi/float(m+1))**2/h**2+\
               sin(YY/2*pi/float(n+1))**2/h**2);
    
    
	b_bar =  dst2(b)       
	u_bar = b_bar/green    
	u = dst2(u_bar)


	return u




