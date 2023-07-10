import numpy as np
from . import a4_IBOF
from . import a4_ORW3
def da2_FT_standard_model(a2,L,CI,rho,a4_method='IBOF'):
    '''a2: the orientation tensor
    rho: the aspect ratio of the fiber
    L: the velocity gradient tensor
    CI: the interaction coefficient
    a4_method: 'IBOF', 'ORW3'
    '''
    I=np.eye(3)
    D=0.5*(L.T+L)
    W=0.5*(L-L.T)
    #import ipdb
    #ipdb.set_trace()
    if a4_method=='IBOF':
        a4=a4_IBOF.a4_IBOF(a2)
    elif a4_method=='ORW3':
        a4=a4_ORW3.a4_ORW3(a2)
    shear_rate=np.sqrt(np.tensordot(D,D,axes=([0,1],[1,0]))*2)
    da2=np.dot(W,a2)-np.dot(a2,W)+ \
            (rho**2-1)/(rho**2+1)*(np.dot(D,a2)+np.dot(a2,D)-2*np.tensordot(a4,D,axes=([2,3],[1,0])))+ \
            2*CI*shear_rate*(I-3*a2)
    return da2
def da2_fun_for_ode(t,a2,CI,L,rho,a4_method='IBOF'):
    '''
    function for ode calculation
    a2 is the array with the shape like (9,)
    t is unused in this function
    rho: aspect ratio
    a2: 2 dimensional orientation tensor
    CI: interaction paramter,0.005<CI<0.05
    a4_method: 'IBOF', 'ORW3'
    
    return: array with the shape of(9,)

    usage:

    from functools import partial
    L=np.zeros((3,3),dtype=np.float64)
    L[0,1]=1
    fun=partial(models.FT_standard_model.da2_fun_for_ode,CI=0.025,L=L,rho=20,a4_method='IBOF')
    a2_0=np.eye(3)/3
    res=integrate.solve_ivp(fun,(0,2000),a2_0.reshape((-1,)),method='RK45')
    '''
    a2=a2.reshape((3,3))
    a2=(a2.T+a2)/2
    da2=da2_FT_standard_model(a2,L,CI,rho,a4_method)
    return da2.reshape((-1,))
