import numpy as np
from . import a4_IBOF
from . import a4_ORW3
def stress_Lipscomb(viscosity_0,L,rho,a2,phi,c1=2):
    '''Lipscomb's method to cal the stress of the fiber suspension 1988 
    L: velocity gradient tensor
    W=0.5*(L-L.T)
    D=0.5*(L+L.T)
    rho:aspect ratio of the fiber
    c1: amaterial constant, c1=2 for rho = +inf
    In dilute region, N=rho^2/(2*ln(rho))
    In semidilute region, N=4rho^2/3*(1/(ln(1/phi)+ln(1/phi)+C'')),
        C''=0.16 when aligned, C''=-0.66 when random orientated
    ref:[1] Aaron P. R. Eberle. et. al. Using transient shear rheology to determine material 
        parameters in fiber suspension theory. Journal of Rheology 53, 685 (2009); 
        doi: 10.1122/1.3099314
    '''
    W=0.5*(L-L.T)
    D=0.5*(L+L.T)
    #a4=a4_ORW3.a4_ORW3(a2)
    a4=a4_IBOF.a4_IBOF(a2)
    if phi<0.01:
        N=rho*rho/(2*np.log(rho))
    else:
        N=4*rho*rho/3*(1/(np.log(1/phi)+np.log(1/phi)+0.16))
    stress=viscosity_0*2*D+ \
            2*c1*phi*viscosity_0*D+ \
            2*phi*viscosity_0*N*np.tensordot(D,a4,axes=([0,1],[1,0]))
    return stress