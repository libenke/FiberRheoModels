import numpy as np

'''velocity gradient tensor L 
for simple shear, planar elongation, center gated disk field
'''
#simple shear
def simple_shear(shear_rate):
    '''return the velocity gradient tensor of the simple shear'''
    L=np.array([[0,0,shear_rate],[0,0,0],[0,0,0]])
    return L
#planar elongation
def planar_elong(elong_rate):
    '''return the velocity gradient tensor of the planar elongation'''
    L=np.array([[elong_rate,0,0],[0,elong_rate,0],[0,0,0]])
    return L
#center gated disk
def center_gated_disk(r,h,z,Q):
    '''
    return the velocity gradient tensor of the center_gated_disk
    r: radius of disk
    h: height of the disk
    z: location in the gap of the disk
    Q: Volume flow rate
    in iARD-RPR model, r=1, h=3, z=1, Q=20.
    '''
    L=3*Q/(8*np.pi*r*h)*np.array([[-1/r*(1-z*z/h/h),0,2/h*z/h],[0,1/r*(1-z*z/h/h),0],[0,0,0]])
    return L
