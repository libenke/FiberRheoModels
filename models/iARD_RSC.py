from . import a4_IBOF
import numpy as np
from scipy import integrate

''' IARD-RSC model'''
def da2_iARD_RSC(a2,CI,CM,kappa,L,rho):
    '''
    dA/dt of the iARD-RSC model.Parameters:
    rho: aspect ratio
    a2: 2 dimensional orientation tensor
    CI: interaction paramter,0.005<CI<0.05; when CI=0, 
    CM: discribe the anisotropic rotation diffusion of the fiber suspension, 0<CM<1.0, CM=0 returns to the IRD term.
    kappa, the RSC term parameter.When kappa=1, the RSC term is removed.
    '''
    I=np.eye(3)
    D=0.5*(L.T+L)
    W=0.5*(L-L.T)
    Dr=CI*(I-CM*(D.dot(D)/np.linalg.norm(D.dot(D),ord=2)))
    a4=a4_IBOF.a4_IBOF(a2)
    #da2_HD
    da2_HD=(np.dot(W,a2)-np.dot(a2,W))+ \
            (rho*rho-1)/(rho*rho+1)*(np.dot(D,a2)+np.dot(a2,D)- \
            2*np.tensordot(a4,D,axes=([2,3],[1,0])))
    #da2_iARD
    shear_rate=np.linalg.norm(L[0,1:3],ord=2)
    da2_iARD=shear_rate*(2*Dr-2*np.trace(Dr)*a2-5*Dr.dot(a2)-5*a2.dot(Dr)+ \
            10*np.tensordot(a4,Dr,axes=([2,3],[1,0])))
    #da2_RSC
    norm_D=np.linalg.norm(D,ord=2)
    if norm_D>1e-4:
        dt=0.0001/norm_D
    else:
        dt=1
    lams,R=np.linalg.eigh(a2)#eigh sort the eigen values by descent
    lams=np.flip(lams,axis=0)
    R=np.flip(R,axis=1)
    lams_iARD_right,_=np.linalg.eigh(a2+dt*(da2_HD+da2_iARD))
    lams_iARD_right=np.flip(lams_iARD_right,axis=0)
    lams_iARD_left,_=np.linalg.eigh(a2-dt*(da2_HD+da2_iARD))
    lams_iARD_left=np.flip(lams_iARD_left,axis=0)
    dlams=(lams_iARD_right-lams_iARD_left)/(2*dt)
    dlam_IOK=np.empty(3)
    #RSC
    dlam_IOK=(1-kappa)*dlams
    da2_RSC=-np.dot(R,np.diag(dlam_IOK)).dot(R.T)
    return da2_HD+da2_iARD+da2_RSC

def ode_fun(t,a2,CI,CM,kappa,L,rho):
    '''
    function for ode calculation
    a2 is the array with the shape like (9,)
    t is unused in this function
    rho: aspect ratio
    a2: 2 dimensional orientation tensor
    CI: interaction paramter,0.005<CI<0.05
    CM: 0<CM<1.0
    kappa: RSC term

    return: array with the shape of(9,)

    usage:

    from functools import partial
    fun=partial(models.iARD_RSC.ode_fun,CI=0.025,CM=1,kappa=0.03,L=models.flow_field.simple_shear(1),rho=30)
    a2_0=np.eye(3)/3
    res=integrate.solve_ivp(fun,(0,2000),a2_0.reshape((-1,)),method='RK45')

    '''
    a2=a2.reshape((3,3))
    a2=(a2.T+a2)/2
    da2=da2_iARD_RSC(a2,CI,CM,kappa,L,rho)
    return da2.reshape((-1,))
