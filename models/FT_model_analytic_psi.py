import numpy as np
def dpsi_pt(t,psi,phi,rho,CI,shear_rate):
    '''
    aspect ratio is added 
    t is unused here
    dΨ/dt=CI*shear_rate*d^2Ψ/dΦ^2-d/dΦ*[Ψ(-sin^2(Φ)*dvx/dvy)]
    =CI*shear_rate*d^2Ψ/dΦ^2- dΨ/dΦ*(-sin^2(Φ)*dvx/dvy)-Ψ*(-2sin(Φ)*dvx/dvy*cos(Φ))
    
    ref:
        F. Folgar, C.L. Tucker III, Orientation behavior of fibers in concentrated suspen- sions, J. Reinf. Plast. Compos. 3 (1984) 98–119.
    usage:
    from functools import partial
    phi=np.arange(-np.pi/2,np.pi/2,np.pi/200,dtype=np.float64)
    # initial state random oriented
    psi_0=np.array([1/np.pi]*len(phi))
    shear_rate=1
    fun=partial(dpsi_pt,phi=phi,CI=0.025,shear_rate=shear_rate)
    tspan=(0,100)
    t_eval=np.linspace(0,tspan[1])
    res=integrate.solve_ivp(fun,(0,200),psi_0,method='RK45',t_eval=t_eval)
    import pandas as pd
    res_df=pd.DataFrame(res['y'],index=phi)
    res_df.index.name='angle'
    res_df.plot()
    '''
    delta_phi=np.mean(np.diff(phi))
    dpsi_dphi=(np.concatenate((psi[1:],psi[0:1]))-np.concatenate((psi[-1:],psi[:-1])))/2/delta_phi
    d2psi_dphi2=(np.concatenate((psi[1:],psi[0:1]))+np.concatenate((psi[-1:],psi[:-1]))-2*psi)/(delta_phi**2)
    dpsi=d2psi_dphi2*CI*shear_rate- \
            dpsi_dphi*(-rho**2/(rho**2+1)*np.sin(phi)**2*shear_rate-1/(rho**2+1)*np.cos(phi)**2*shear_rate)- \
            psi*(rho**2-1)/(rho**2+1)*(-2)*np.sin(phi)*shear_rate*np.cos(phi)
    return dpsi


def dphi_dt(t,phi,rho,psi_stable_interp_fun,CI,shear_rate):
    '''
    calculate the dΦ/dt
    ref: 
        F. Folgar, C.L. Tucker III, Orientation behavior of fibers in concentrated suspen- sions, J. Reinf. Plast. Compos. 3 (1984) 98–119.
    usage:
    from functools import partial
    phi_span=np.arange(-np.pi/2,np.pi/2,np.pi/500,dtype=np.float64)
    rho=20
    
    # calculate psi_stable
    psi_0=np.array([1/np.pi]*len(phi_span))
    shear_rate=1
    fun=partial(dpsi_pt,phi=phi_span,rho=rho,CI=0.025,shear_rate=shear_rate)
    tspan=(0,100)
    t_eval=np.linspace(0,tspan[1])
    res=integrate.solve_ivp(fun,tspan,psi_0,method='RK45',t_eval=t_eval)
    psi_stable=res['y'][:,-1]
    
    #interpolate the Ψ(Φ) in region [-1.5pi,1.5pi)    
    from scipy.interpolate import interp1d
    phi_span_interp=np.concatenate((phi_span-np.pi,phi_span,phi_span+np.pi))
    psi_stable_interp=np.concatenate((psi_stable,psi_stable,psi_stable))
    psi_stable_interp_fun=interp1d(phi_span_interp,psi_stable_interp,kind='cubic')

    # calculate the rotation of the fiber
    rotation_fun=partial(dphi_dt,rho=rho,psi_stable_interp_fun=psi_stable_interp_fun,CI=0.025,shear_rate=shear_rate)
    tspan=(0,100)
    #t_eval=np.linspace(0,tspan[1],num=2000)
    phi_0=np.array([0,])
    res=integrate.solve_ivp(rotation_fun,tspan,phi_0,method='LSODA',max_step=0.2)
    phi=res['y'][0,:]
    phi=(phi+np.pi/2)%np.pi-np.pi/2
    df=pd.DataFrame(phi,index=res['t'])
    df.plot()
    '''
    #cast the phi into the range [-pi/2 , pi/2)
    phi=(phi+np.pi/2)%np.pi-np.pi/2
    #dΨ/dΦ
    delta_phi=0.0002
    dpsi_dphi=(psi_stable_interp_fun(phi+delta_phi)-psi_stable_interp_fun(phi-delta_phi))/(2*delta_phi)
    #dΦ/dt
    #dphi=-np.sin(phi)**2*shear_rate- \
    #       CI*shear_rate/psi_stable_interp_fun(phi)*dpsi_dphi
    dphi=-rho**2/(rho**2+1)*np.sin(phi)**2*shear_rate- \
            1/(rho**2+1)*np.cos(phi)**2*shear_rate- \
            CI*shear_rate/psi_stable_interp_fun(phi)*dpsi_dphi
    return dphi
