import pandas as pd
from scipy import integrate
import numpy as np
import pyshtools as sht
import pyshtools
from .ODF_Sph_Harm import ODF_Sph_Harm
from scipy import sparse
import scipy

class ODF_Sph_Harm_Confined(ODF_Sph_Harm):
    '''solve the Confined model using the Spherical Harmonics functions'''
    def __repr__(self):
        return  "iARD model with parameters:\nCI="+str(self.CI)+";\n" \
			  "CM="+str(self.CM)+";\nshear_rate="+str(self.shear_rate)+";\n" \
			  "degree="+str(self.degree)+";\nt_max="+str(np.max(self.t_span))+";\nmax_dt="+str(self.max_dt)+"."
    def __init__(self,CI,CM,rho,k=1,confined_epsilon=0.04,confined_alpha=0.5,shear_rate=1,degree=100,init_sphharm_co=None, \
			max_dt=0.02,t_span=np.logspace(-2,2,100)):
        ''' rho: aspect ratio
        a2: 2 dimensional orientation tensor
        CI: interaction paramter,0.005<CI<0.05
        CM: 0<CM<1.0
        k=1-alpha, k=1 respresent no slow down of the fibers; k=0.033 represent 30 times slow down
        degree: the number of degree of spherical harmonic functions to use
        initial_sphharm_co:  the initial state of the spherical harmonic functions. 
                                    with the shape of :[degree+1, 2*degree+1] 
        max_dt: maximum of the interval time
        '''
        # initialize the super class: ODF_Sph_Harm
        ODF_Sph_Harm.__init__(self,CI,CM,rho,k=1,shear_rate=shear_rate,degree=degree,init_sphharm_co=None, \
			max_dt=max_dt,t_span=t_span)
        self.confined_epsilon=confined_epsilon
        self.confined_alpha=confined_alpha
    def Confined_Term_matrix_form(self):
        '''My Confined term,type 1 of the confinement model:
        1 is the virtual compression flow field D_star
        2 is the multiply the affine deformation part with (1+ε)
        D_star=ε[[0.5+alpha,0,0],[0,-1,0],[0,0,0.5-alpha]]
        Confined_term=-∇∙[ξ(D_star∙p-D_star:ppp)ψ]
        usage:
            input:eps, the compress rate, alpha, the finetune parameter
        return:
            the div confined term, the operator matrix of the spherical harmonics coefficients represent the confined term
        '''
        eps=self.confined_epsilon
        alpha=self.confined_alpha
        shear_rate=self.shear_rate
        x=self.x_matrix
        y=self.y_matrix
        z=self.z_matrix
        xi=self.xi
        deltax=self.deltax_matrix
        deltay=self.deltay_matrix
        deltaz=self.deltaz_matrix
        # calculate the ξ(D_star∙p-D_star:ppp) ψ term
        
        confined_psi=[0.5*eps*x + alpha*eps*x - 0.5*eps*x.dot(x.dot(x)) - alpha*eps*x.dot(x.dot(x)) + \
               eps*x.dot(y.dot(y)) - 0.5*eps*x.dot(z.dot(z)) + alpha*eps*x.dot(z.dot(z)), \
               -eps*y - 0.5*eps*x.dot(x).dot(y) - alpha*eps*x.dot(x).dot(y)+eps*y.dot(y.dot(y)) - \
               0.5*eps*y.dot(z.dot(z)) + alpha*eps*y.dot(z.dot(z)),  \
               0.5*eps*z - alpha*eps*z - 0.5*eps*x.dot(x)*z - alpha*eps*x.dot(x).dot(z) + \
               eps*y.dot(y).dot(z) - 0.5*eps*z.dot(z.dot(z)) + alpha*eps*z.dot(z.dot(z))]
        div_confined_matrix=-(deltax.dot(confined_psi[0])+deltay.dot(confined_psi[1])+deltaz.dot(confined_psi[2]))
        return xi*div_confined_matrix
    
    def run_confined_matrix_type(self,dt=0.001):
        '''My Confined term,type 1 of the confinement model:
        1 is the virtual compression flow field D_star
        2 is the multiply the affine deformation part with (1+ε)'''
        psi_co_flatten=self.init_sphharm_co.flatten()
        div_matrix=self.HD_Term_matrix_form()[0]+self.iARD_Term_matrix_form()[0]+ \
                    self.Confined_Term_matrix_form()
        res_t=[]
        res_y=[]
        A2=[]
        t_max=np.max(self.t_span)
        num=len(self.t_span)
        interv=int(t_max/dt/num)
        t_steps=np.arange(0,t_max,dt)
        for n_i,t in enumerate(t_steps):
            dpsi=div_matrix.dot(psi_co_flatten)
            
            psi_co_flatten+=dpsi*dt
            if np.mod(n_i,interv)==0:
                #print(t)
                res_t.append(t)
                clm_co=np.reshape(psi_co_flatten,[self.degree+1,2*self.degree+1])
                res_y.append(clm_co.copy())
                A2.append(self.get_Orientation_Tensor_A2(clm_co).flatten())
        res={'time_span':np.array(res_t), \
             'sphharm_co':np.array(res_y), \
             'A2':np.array(A2), \
             'CI': self.CI, \
             'CM': self.CM, \
             'eps': self.confined_epsilon, \
             'alpha': self.confined_alpha, \
             'rho': self.rho, \
             'degree': self.degree, \
             'shear_rate': self.shear_rate, \
             'init_sphharm_co':self.init_sphharm_co}
        return res
    
    def cal_single_fiber_orbit_confined(self,p0,psi_clm,max_time=100,num=1000,max_step=0.05):
        '''calculate the trajetory of single fiber predicted by the type 1 of the confined model.
        My Confined term,type 1 of the confinement model:
        1 is the virtual compression flow field D_star
        2 is the multiply the affine deformation part with (1+ε)
        usage:
        input: p0:the unit orientation vector at time=0;
                psi_clm: the spherical harmonics coefficient object (Coeffi) provided by pyshtools packages;
                max_time: I always set the shear rate ==0 , so the max time means the max strain to be calculated
                num: the sampling number of the trajetory of single fiber
                max_step: the maximum step during calculate the tracjetory used in RK45 method
        
        return: a DataFrame object contains trajetory of single fiber
                the index is the time; and the columns is ['x','y','z','theta','phi']
        note in this function:
            the + - of the W.p term is very important, that it will influence the period behavior!
            just becareful if should modify this function again!
            if this was wrong, it will not rotate numericallly!'''
        #calculate the spherica harmonics coefficient of the psi 
        psi_grid=psi_clm.expand(grid='DH')
        # calculate the ln(psi)
        ln_psi_grid=psi_grid.copy()
        ln_psi_grid.data=np.log(np.real(psi_grid.data))
        ln_psi_clm=ln_psi_grid.expand()
        #calculate the \nabla ln(psi)
        nabla_ln_psi_clm_x=self.to_clm(self.deltax(self.to_clm_co(ln_psi_clm)))
        nabla_ln_psi_clm_y=self.to_clm(self.deltay(self.to_clm_co(ln_psi_clm)))
        nabla_ln_psi_clm_z=self.to_clm(self.deltaz(self.to_clm_co(ln_psi_clm)))
        # def the function for ode to calculate the unit vector
        def fun_for_ode(t,p):
            #the pyshtools using the lons and lats, we should transform the phi theta to lats and lons
            p=p/np.linalg.norm(p,ord=2)
            #calculate the theta phi and lats, lons
            x=p[0]
            y=np.array(p[1])
            z=np.array(p[2])
            theta=np.arccos(z)
            #this step is very important!!!!
            if x>=0:
                phi=np.arctan(y/x)
            else:
                phi=(np.arctan(y/x)+np.pi)
            lat=float(-(theta-np.pi/2)/np.pi*180)
            lon=float(np.mod(-(phi-np.pi)/np.pi*180,360))
            #calculate the parameters
            L=np.array([[0,self.shear_rate,0],[0,0,0],[0,0,0]])
            D=0.5*(L.T+L)+self.confined_epsilon*(np.array([[0.5+self.confined_alpha,0,0],[0,-1,0],[0,0,0.5-self.confined_alpha]]))
            W=0.5*(L-L.T)
            shear_rate=np.sqrt(2*np.tensordot(D,D,axes=[[0,1],[1,0]]))
            I=np.eye(3)
            pp=np.tensordot(p,p,axes=0)
            #calculate dot_p_HD term
            dot_p_HD=np.tensordot(W,p,axes=[1,0])+  \
                     self.xi*(np.tensordot(D,p,axes=[1,0])- \
                      np.tensordot(D,pp,axes=[[1,0],[0,1]])*p)
            #calculate the (1-pp).dr.(I-pp) term
            dr=self.CI*(I-self.CM*np.dot(D,D)/np.linalg.norm(np.dot(D,D),ord=2))
            I=np.eye(3)
            Ipp_dr_Ipp=np.dot(np.dot((I-pp),dr),I-pp)
            #calculate the nabla ln(psi) value
            nabla_ln_psi=np.array([nabla_ln_psi_clm_x.expand(grid='DH',lat=lat,lon=lon), \
                                   nabla_ln_psi_clm_y.expand(grid='DH',lat=lat,lon=lon), \
                                   nabla_ln_psi_clm_z.expand(grid='DH',lat=lat,lon=lon)])
            #calculate the shear_rate*Ipp_dr_Ipp_∇ln(Ψ) term
            Ipp_dr_Ipp_nabla_ln_psi=np.real(np.abs(shear_rate)*np.dot(Ipp_dr_Ipp,nabla_ln_psi))
            return dot_p_HD-Ipp_dr_Ipp_nabla_ln_psi
        #perform the RK45 method to caulculate the evolution of p
        t_span=[0,max_time*1.01]
        t_eval=np.linspace(0,max_time,num)
        res=integrate.solve_ivp(fun_for_ode,t_span,p0,t_eval=t_eval,method='RK45',rtol=1e-4,atol=1e-7,max_step=max_step)
        df=pd.DataFrame(res.y.T,index=res.t,columns=['x','y','z'])
        df.loc[:,'theta']=np.arccos(df['z'])
        df.loc[:,'phi']=np.arctan(df['y']/df['x'])
        return df

    def Confined_Term_matrix_form_2(self):
        '''My Confined term,type 2 of the confinement model:
        1 is the virtual compression flow field D_star
        2 is the multiply the affine deformation part with (1+ε)
        Confined_term=-ε*∇∙[(D∙p-D:ppp)ψ]
        usage:
            input:eps, the compress rate, alpha, the finetune parameter
        return:
            the div confined term, the operator matrix of the spherical harmonics coefficients represent the confined term
        '''
        eps=self.confined_epsilon
        shear_rate=self.shear_rate
        x=self.x_matrix
        y=self.y_matrix
        z=self.z_matrix
        xi=self.xi
        deltax=self.deltax_matrix
        deltay=self.deltay_matrix
        deltaz=self.deltaz_matrix
        # calculate the ξ(D_star∙p-D_star:ppp) ψ term
        # calculate the dotp ψ term
        dot_p=[xi*shear_rate/2*y-xi*shear_rate*x.dot(x.dot(y)), \
               xi*shear_rate/2*x-xi*shear_rate*x.dot(y.dot(y)),  \
               -xi*np.abs(shear_rate)*x.dot(y.dot(z))]
        div_HD_matrix=-eps*(deltax.dot(dot_p[0])+deltay.dot(dot_p[1])+deltaz.dot(dot_p[2]))
        return self.revmove_zeros(div_HD_matrix)
    
    def run_confined_matrix_type_2(self,dt=0.001):
        '''My Confined term,type 2 of the confinement model:
        1 is the virtual compression flow field D_star
        2 is the multiply the affine deformation part with (1+ε)'''
        psi_co_flatten=self.init_sphharm_co.flatten()
        div_matrix=self.HD_Term_matrix_form()[0]+self.iARD_Term_matrix_form()[0]+ \
                    self.Confined_Term_matrix_form_2()
        res_t=[]
        res_y=[]
        A2=[]
        t_max=np.max(self.t_span)
        num=len(self.t_span)
        interv=int(t_max/dt/num)
        t_steps=np.arange(0,t_max,dt)
        for n_i,t in enumerate(t_steps):
            dpsi=div_matrix.dot(psi_co_flatten)
            
            psi_co_flatten+=dpsi*dt
            if np.mod(n_i,interv)==0:
                #print(t)
                res_t.append(t)
                clm_co=np.reshape(psi_co_flatten,[self.degree+1,2*self.degree+1])
                res_y.append(clm_co.copy())
                A2.append(self.get_Orientation_Tensor_A2(clm_co).flatten())
        res={'time_span':np.array(res_t), \
             'sphharm_co':np.array(res_y), \
             'A2':np.array(A2), \
             'CI': self.CI, \
             'CM': self.CM, \
             'eps': self.confined_epsilon, \
             'rho': self.rho, \
             'degree': self.degree, \
             'shear_rate': self.shear_rate, \
             'init_sphharm_co':self.init_sphharm_co}
        return res
    
    def cal_single_fiber_orbit_confined_2(self,p0,psi_clm,max_time=100,num=1000,max_step=0.05):
        '''calculate the trajetory of single fiber predicted by the model.
        usage:
        input: p0:the unit orientation vector at time=0;
                psi_clm: the spherical harmonics coefficient object (Coeffi) provided by pyshtools packages;
                max_time: I always set the shear rate ==0 , so the max time means the max strain to be calculated
                num: the sampling number of the trajetory of single fiber
                max_step: the maximum step during calculate the tracjetory used in RK45 method
        
        return: a DataFrame object contains trajetory of single fiber
                the index is the time; and the columns is ['x','y','z','theta','phi']
        note in this function:
            the + - of the W.p term is very important, that it will influence the period behavior!
            just becareful if should modify this function again!
            if this was wrong, it will not rotate numericallly!'''
        #calculate the spherica harmonics coefficient of the psi 
        psi_grid=psi_clm.expand(grid='DH')
        # calculate the ln(psi)
        ln_psi_grid=psi_grid.copy()
        ln_psi_grid.data=np.log(np.real(psi_grid.data))
        ln_psi_clm=ln_psi_grid.expand()
        #calculate the \nabla ln(psi)
        nabla_ln_psi_clm_x=self.to_clm(self.deltax(self.to_clm_co(ln_psi_clm)))
        nabla_ln_psi_clm_y=self.to_clm(self.deltay(self.to_clm_co(ln_psi_clm)))
        nabla_ln_psi_clm_z=self.to_clm(self.deltaz(self.to_clm_co(ln_psi_clm)))
        # def the function for ode to calculate the unit vector
        def fun_for_ode(t,p):
            #the pyshtools using the lons and lats, we should transform the phi theta to lats and lons
            p=p/np.linalg.norm(p,ord=2)
            #calculate the theta phi and lats, lons
            x=p[0]
            y=np.array(p[1])
            z=np.array(p[2])
            theta=np.arccos(z)
            #this step is very important!!!!
            if x>=0:
                phi=np.arctan(y/x)
            else:
                phi=(np.arctan(y/x)+np.pi)
            lat=float(-(theta-np.pi/2)/np.pi*180)
            lon=float(np.mod(-(phi-np.pi)/np.pi*180,360))
            #calculate the parameters
            L=np.array([[0,self.shear_rate,0],[0,0,0],[0,0,0]])
            ####################################################
            #the difference of the two models are shown in D
            #here is the type of multy A with (1+ε)
            D=0.5*(L.T+L)*(1+self.confined_epsilon)
            ####################################################
            W=0.5*(L-L.T)
            shear_rate=np.sqrt(2*np.tensordot(D,D,axes=[[0,1],[1,0]]))
            I=np.eye(3)
            pp=np.tensordot(p,p,axes=0)
            #calculate dot_p_HD term
            dot_p_HD=np.tensordot(W,p,axes=[1,0])+  \
                     self.xi*(np.tensordot(D,p,axes=[1,0])- \
                      np.tensordot(D,pp,axes=[[1,0],[0,1]])*p)
            #calculate the (1-pp).dr.(I-pp) term
            dr=self.CI*(I-self.CM*np.dot(D,D)/np.linalg.norm(np.dot(D,D),ord=2))
            I=np.eye(3)
            Ipp_dr_Ipp=np.dot(np.dot((I-pp),dr),I-pp)
            #calculate the nabla ln(psi) value
            nabla_ln_psi=np.array([nabla_ln_psi_clm_x.expand(grid='DH',lat=lat,lon=lon), \
                                   nabla_ln_psi_clm_y.expand(grid='DH',lat=lat,lon=lon), \
                                   nabla_ln_psi_clm_z.expand(grid='DH',lat=lat,lon=lon)])
            #calculate the shear_rate*Ipp_dr_Ipp_∇ln(Ψ) term
            Ipp_dr_Ipp_nabla_ln_psi=np.real(np.abs(shear_rate)*np.dot(Ipp_dr_Ipp,nabla_ln_psi))
            return dot_p_HD-Ipp_dr_Ipp_nabla_ln_psi
        #perform the RK45 method to caulculate the evolution of p
        t_span=[0,max_time*1.01]
        t_eval=np.linspace(0,max_time,num)
        res=integrate.solve_ivp(fun_for_ode,t_span,p0,t_eval=t_eval,method='RK45',rtol=1e-4,atol=1e-7,max_step=max_step)
        df=pd.DataFrame(res.y.T,index=res.t,columns=['x','y','z'])
        df.loc[:,'theta']=np.arccos(df['z'])
        df.loc[:,'phi']=np.arctan(df['y']/df['x'])
        return df
    def cal_period_strain_map(self,psi_clm,theta_num=20,max_strain=300,sampling_num=5000):
        ''' change the initial angle of the fiber, calculate the trajetory of the fiber. And return the period strain map.
        uasage: 
        input:  psi_clm: Spherical Harmonics coeefficients object of the psi,
                theta_num, the number-1 of the thetas to be calcualted
        return a tupule with data:
            0: pandas DataFrame，the rotation period strain of single fiber with the initial theta phi
            1: flattened rotation period strain for 3D plotting
            2: average of the period strain by guassian Legendre quadrature mehod'''
        #the period strain will be stored in the period_df DataFrame 
        period_df=pd.DataFrame()
        period_df.index.name='theta'
        period_df.columns.name='phi'
        from scipy.signal import find_peaks
        #select the intial theta and phi to calculate the period strain
        #I choose the 'GLQ' grid provide by pyshtools, which means 'Guassian Legendre Quadrature'
        #the weithts are also provided by the psi_grid.weights property
        psi_grid=psi_clm.expand(grid='GLQ',lmax=theta_num)
        thetas=-psi_grid.lats()/180*np.pi+np.pi/2
        phis=-psi_grid.lons()/180*np.pi+np.pi
        # calculate period strain in different theta and phi
        for theta in thetas[:int(np.floor(theta_num/2+1))]:
            for phi in phis:
                p0=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
                df=self.cal_single_fiber_orbit(p0,psi_clm,max_strain,num=sampling_num,max_step=0.5)
                p,_=find_peaks(np.abs(df.loc[:,'phi']))
                period_df.loc[theta,phi]=2*(p[1]-p[0])/sampling_num*max_strain
        # due to we only calculate half of the thetaes, so we should add the symmetric part about theta
        if theta_num%2==0:
            for theta in period_df.index[:-1]:
                period_df.loc[np.pi-theta,:]=period_df.loc[theta,:]
        else:
            for theta in period_df.index:
                period_df.loc[np.pi-theta,:]=period_df.loc[theta,:]
        period_df=period_df.sort_index()
        #flatten type of period_df, used for 3D plotting;
        #usage: from mpl_toolkits.mplot3d import axes3d
        #ax = pylab.gca(projection='3d')
        #ax.plot(period_df_flatten.theta,period_df_flatten.phi,period_df_flatten.period_strain)
        theta_span,phi_span=np.meshgrid(period_df.index,period_df.columns)
        flatten_data=np.array(period_df).flatten()
        period_df_flatten=pd.DataFrame([theta_span.T.flatten(),phi_span.T.flatten(),flatten_data], \
                           index=['theta','phi','period_strain']).T
        #calculate the average of the period strain of single fiber using Guassian Legendre quadrature
        df_period_psi=period_df.copy()*np.real(psi_grid.data)
        Legendre_quadrature_weights=pd.DataFrame(np.tile(psi_grid.weights,[len(phis),1]).T)
        df_period_psi*=Legendre_quadrature_weights
        period_average=df_period_psi.sum().sum()
        
        return (period_df,period_df_flatten,period_average)
