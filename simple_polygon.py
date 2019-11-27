
from numpy import squeeze, array,diff, min, max, sum, sqrt,\
                  logical_and, logical_or, ones, zeros, take, arctan2, sin, cos
import scipy.io as sio

class SimplePolygon(object):
    def __init__(self, filename_chm):
        
        if type(filename_chm)==str:
            dict_chm=sio.loadmat(filename_chm)
        else:
            dict_chm=filename_chm

        Vx=squeeze(dict_chm['Vx'])
        Vy=squeeze(dict_chm['Vy'])
        

        
        self.N_vert=len(Vx)
        
        N_edg=len(Vx)

        Vx=list(Vx)
        Vy=list(Vy)
        
        Vx.append(Vx[0])
        Vy.append(Vy[0])
        
        Vx=array(Vx)
        Vy=array(Vy)
        
        self.Vx=Vx
        self.Vy=Vy
        self.N_edg=N_edg
        self.x_min = min(self.Vx)
        self.x_max = max(self.Vx)
        self.y_min = min(self.Vy)
        self.y_max = max(self.Vy)

        self.N_mp_impact=0
        self.N_mp_corrected=0
        self.chamb_type='simple_polyg'
    
    
    def is_outside(self, x_mp, y_mp):

        x_mp_chk=x_mp
        y_mp_chk=y_mp
        N_pts=len(x_mp_chk)
        flag_inside_chk=array(N_pts*[True])
        for ii in range(self.N_edg):
            flag_inside_chk[flag_inside_chk]=((y_mp_chk[flag_inside_chk]-self.Vy[ii])*(self.Vx[ii+1]-self.Vx[ii])\
                                            -(x_mp_chk[flag_inside_chk]-self.Vx[ii])*(self.Vy[ii+1]-self.Vy[ii]))>0
        flag_outside=~flag_inside_chk
            
        return flag_outside
    
   
