import numpy as np
from . import a4_IBOF
from . import a4_ORW3

def da2_Férec(a2,L,k,phi,q,Yc=1/6):
    '''L: velocity gradient tensor
    W=0.5*(L-L.T)
    D=0.5*(L.T+L)
    phi: volume concentration of the fiber suspension
    q: interaction coefficient
    b2=3Pi/8(a2-a4:a2)
    f=trace(b2)
    b4=1/f(b2b2)
    Xc,Yc are resistance function
    Xc=4Pi/r^2, Yc=1/6 for dilute regime
    k: intensity of friction force
    M=2k/(3PiYc)
    shear_rate=sqrt(D:D/2)
    '''
    M=2*k/(3*np.pi*Yc)
    W=0.5*(L-L.T)
    D=0.5*(L.T+L)
    a4=a4_IBOF.a4_IBOF(a2)
    #a4=a4_ORW3.a4_ORW3(a2)
    b2=3*np.pi/8*(a2-np.tensordot(a4,a2,axes=([2,3],[1,0])))
    f=np.trace(b2)
    I=np.eye(3)
    b4=1/f*np.tensordot(b2,b2,axes=0)
    shear_rate=np.sqrt(np.tensordot(D,D,axes=([0,1],[1,0]))*2)
    da2=0.5*(np.dot(2*W,a2)-np.dot(a2,2*W))+ \
            0.5*(np.dot(2*D,a2)+np.dot(a2,2*D)-2*np.tensordot(2*D,a4,axes=([0,1],[1,0])))- \
            0.5*phi*M*(np.dot(2*D,b2)+np.dot(b2,2*D)-2*np.tensordot(2*D,b4,axes=([0,1],[1,0])))+ \
            2*f*phi*M*q*shear_rate*(I-3*a2)
    return da2

def stress(viscosity_0,L,rho,phi,k,a2,XA=1):
    '''L: velocity gradient tensor
    W=0.5*(L-L.T)
    D=0.5*(L.T+L)
    k: intensity of friction force
    XA=1
    rho:aspect ratio of the fiber
    '''
    W=0.5*(L-L.T)
    D=0.5*(L.T+L)
    a4=a4_IBOF.a4_IBOF(a2)
    #a4=a4_ORW3.a4_ORW3(a2)
    b2=3*np.pi/8*(a2-np.tensordot(a4,a2,axes=([2,3],[1,0])))
    f=np.trace(b2)
    b4=1/f*np.tensordot(b2,b2,axes=0)
    stress=viscosity_0*2*D+ \
            viscosity_0*phi*rho*rho/6/np.pi*XA*np.tensordot(2*D,a4,axes=([0,1],[1,0]))+ \
            viscosity_0*phi*phi*4*rho*rho/3/np.pi**2*k*np.tensordot(2*D,b4,axes=([0,1],[1,0]))
    return stress

def da2_fun_for_ode(t,a2,L,k,phi,q,Yc=1/6):
    '''
    function for ode calculation
    a2 is an array with the shape like (9,)
    t is unused in this function
    a2: 2 dimensional orientation tensor
    L: velocity gradient tensor
    Xc,Yc are resistance function
    Xc=4Pi/r^2, Yc=1/6 for dilute regime
    k: intensity of friction force
    phi: volume concentration of the fiber suspension
    q: interaction coefficient
    return: array with the shape of(9,)

    usage:

    from functools import partial
    fun=partial(models.Férec.da2_fun_for_ode,L=models.flow_field.simple_shear(1),k=0.1,phi=0.1,q=0.22,Yc=1/6)
    a2_0=np.eye(3)/3
    res=integrate.solve_ivp(fun,(0,2000),a2_0.reshape((-1,)),method='RK45')

    '''
    a2=a2.reshape((3,3))
    a2=(a2.T+a2)/2
    da2=da2_Férec(a2,L,k,phi,q,Yc)
    return da2.reshape((-1,))
