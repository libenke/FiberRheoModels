import pandas as pd
from scipy import integrate
import numpy as np
import pyshtools as sht
import pyshtools
from . import sph_harm_operator
from scipy import sparse
import scipy
sph_harm_operator=sph_harm_operator.sph_harm_operator

class ODF_Sph_Harm(sph_harm_operator):
    '''solve the FT_iARD_RSC model using the Spherical Harmonics functions alone'''
    def __repr__(self):
        return  "iARD model with parameters:\nCI="+str(self.CI)+";\n" \
			  "CM="+str(self.CM)+";\nshear_rate="+str(self.shear_rate)+";\n" \
			  "degree="+str(self.degree)+";\nt_max="+str(np.max(self.t_span))+";\nmax_dt="+str(self.max_dt)+"."
    def __init__(self,CI,CM,rho,k=1,shear_rate=1,degree=100,init_sphharm_co=None, \
			max_dt=0.02,t_span=np.linspace(0,100,50)):
        ''' Solve the FT iARD RSC model alone using the Spherical Harmonic method alone.
		rho: aspect ratio
        a2: 2 dimensional orientation tensor
        CI: interaction paramter,0.005<CI<0.05
        CM: 0<CM<1.0
        k=1-alpha, k=1 respresent no slow down of the fibers; k=0.033 represent 30 times slow down
        degree: the number of degree of spherical harmonic functions to use
        initial_sphharm_co:  the initial state of the spherical harmonic functions. 
                                    with the shape of :[degree+1, 2*degree+1] 
        max_dt: maximum of the interval time
        '''
        # initialize the super class: provide the spherical harmonics operators
        sph_harm_operator.__init__(self,degree)
        #the spherical harmonics parameters
        if init_sphharm_co is None:
            clm=sht.SHCoeffs.from_zeros(degree,kind='complex',normalization='4pi')
            clm.set_coeffs(1/4/np.pi,0,0)
            self.init_sphharm_co=self.to_clm_co(clm)
        else:
            self.init_sphharm_co=init_sphharm_co
            clm=self.to_clm(init_sphharm_co)
        self.init_psi_grid=clm.expand(grid='DH')
        # change the lats and lons to the theta and phi
        self.theta_span=-self.init_psi_grid.lats()/180*np.pi+np.pi/2
        self.phi_span=self.init_psi_grid.lons()/180*np.pi-np.pi
        theta_mesh,phi_mesh=np.meshgrid(self.theta_span,self.phi_span,indexing='ij')        
        self.theta_mesh=theta_mesh
        self.phi_mesh=phi_mesh
        #set the parameters of the model
        self.CI=CI
        self.CM=CM
        self.rho=rho
        self.xi=(rho**2-1)/(rho**2+1)
        self.shear_rate=shear_rate
        self.t_span=t_span
        self.max_dt=max_dt
        self.k=k
        rand_clm=pyshtools.SHCoeffs.from_random(np.random.rand(degree+1),kind='complex')
        self.rand_clm_co=self.to_clm_co(rand_clm)
        #
    def run(self):
        '''use the RK45 method to eval the results
        return a dict: including (time_span, sphharm_co_array, psi_grid_array, A2_dataframe)
        sphharm_co_array's dimensions: (:, degree+1, 2*degree+1)
        '''
        t_eval=self.t_span
        t_span=[0,np.max(t_eval)+0.01]
        # wrap the dpsi_co_dt function for ode solver
        num_t=0
        def fun(t,psi_co_1d):
            #nonlocal num_t
            #num_t+=1
            #if np.mod(num_t,200)==0:
            #    print(t)
            psi_co_2d=np.reshape(psi_co_1d,self.init_sphharm_co.shape)
            return np.reshape(self.dpsi_co_dt(psi_co_2d),[-1])
        #run the initial value problem using the RK45
        y0=np.reshape(self.init_sphharm_co,[-1])
        res=integrate.solve_ivp(fun, t_span, y0, \
				method='RK23',t_eval=t_eval,max_step=self.max_dt)
        # the spherical harmonics functions' coefficient array
        res_sphharm_co=np.reshape(res.y,[self.degree+1,2*self.degree+1,-1])
        #calculate the A2 tensor 
        A2_list=[]
        psi_grid=[]
        for ind,t in enumerate(res.t):
            co=res_sphharm_co[:,:,ind]
            A2=self.get_Orientation_Tensor_A2(co)
            A2_list.append(np.reshape(A2,[-1]))
            psi_grid.append(self.to_clm(co).expand(grid='DH').data)
        return {'time_span': res.t, \
                'A2': np.array(A2_list), \
                'psi_grid': np.array(psi_grid), \
                'sphharm_co': res_sphharm_co, \
                'theta_span':self.theta_span, \
                'phi_span': self.phi_span, \
                'theta_mesh':self.theta_mesh, \
                'phi_mesh':self.phi_mesh, \
                'CI':self.CI, \
                'CM':self.CM, \
                'rho':self.rho, \
                'k': 0, \
                'shear_rate':self.shear_rate, \
                'degree':self.degree, \
                'init_sphharm_co':self.init_sphharm_co, \
                'init_psi_grid':self.init_psi_grid}
    def dpsi_co_dt(self,psi_co):
        '''iARD model only for simple shear
        input psi: the coefficients of the spherical harmonics
        usage:
        degree=100
        psi=sht.SHCoeffs.from_zeros(degree,kind='complex')
        psi.set_coeffs(1/4/np.pi,0,0)
        psi=np.concatenate((np.fliplr(clm.coeffs[1,:,1:]),clm.coeffs[0,:,:]),axis=1)
        dpsi_dt(clm_co)

        # the HD term:
        D=[[0, shear_rate/2,0],[0,0,0],[shear_rate/2,0,0]]
        W=[[0, shear_rate/2,0],[0,0,0],[-shear_rate/2,0,0]]

        dot_p=W•u+ξ(D•u-D:uuu)
        =
        [(1+ξ)*shearrate/2*y - ξ*shearrate*x*x*y);
         (ξ-1)*shearrate/2*x - ξ*shearrate*x*y*y);
         -ξ*shearrate*x*y*z]'''
        #list the parameters of the model here
        shear_rate=self.shear_rate
        CM=self.CM
        CI=self.CI
        psi=psi_co
        x=self.x
        y=self.y
        z=self.z
        xi=self.xi
        # calculate the dotp ψ term
        dot_p_psi=[(1+xi)*shear_rate/2*y(psi)-xi*shear_rate*x(x(y(psi))), \
                   (xi-1)*shear_rate/2*x(psi)-xi*shear_rate*x(y(y(psi))),  \
                   -xi*shear_rate*x(y(z(psi)))]
        '''calculate the iARD term:
        Dr=[[CI-CI*CM,         0,          0],
              [0,                CI-CI*CM,   0],
              [0,                     0,          CI]]
        Grad=[∇_xf, ∇_yf, ∇_zf]
        (I-uu)•Dr•(I-uu)•∇_s(f)=(I-uu)•Dr•∇_s(f)=(I-uu)•(Dr•∇_s(f))
        Dr•∇_s(f)=
        [(CI-CI*CM)*∇_xf,(CI-CI*CM)*∇_yf,CI*∇_zf]'''
        surf_grad_psi=self.surface_gradient(psi)
        dr_dot_surfgrad_psi=[(CI-CI*CM)*surf_grad_psi[0],
                                 (CI-CI*CM)*surf_grad_psi[1],
                                 CI*surf_grad_psi[2]]
        iARD_psi=self.I_uu_dot(dr_dot_surfgrad_psi)
        iARD_psi=[-iARD_psi[0]*shear_rate,-iARD_psi[1]*shear_rate,-iARD_psi[2]*shear_rate]
        flux=[dot_p_psi[0]+iARD_psi[0],dot_p_psi[1]+iARD_psi[1],dot_p_psi[2]+iARD_psi[2]]
        return -self.divergence(flux)
    
    def HD_Term_matrix_form(self):
        '''
        the HD term.
        write the operators as matrix from, and unite all the operators to one final operator matrix,
        wicth can be act on the flattened spherical coefficients
        return: a tuple, 
            the first value is the operator matrix represent the Jeffery HD term: div.dot_p
                with the size of [(degree+1)*(2*degree+1), (degree+1)*(2*degree+1)], 
            the second is the dot_p term which is used for calculate single fiber's orbit
        '''
        shear_rate=self.shear_rate
        x=self.x_matrix
        y=self.y_matrix
        z=self.z_matrix
        xi=self.xi
        deltax=self.deltax_matrix
        deltay=self.deltay_matrix
        deltaz=self.deltaz_matrix
        I_uu_op=self.I_uu_dot_matrix_op
        surf_deltax,surf_deltay,surf_deltaz=I_uu_op([deltax,deltay,deltaz])
        # calculate the dotp ψ term
        dot_p=[(1+xi)*shear_rate/2*y-xi*shear_rate*x.dot(x.dot(y)), \
               (xi-1)*shear_rate/2*x-xi*shear_rate*x.dot(y.dot(y)),  \
               -xi*shear_rate*x.dot(y.dot(z))]
        surf_div_HD_matrix=-(surf_deltax.dot(dot_p[0])+surf_deltay.dot(dot_p[1])+surf_deltaz.dot(dot_p[2]))
        return (self.revmove_zeros(surf_div_HD_matrix),dot_p)
    
    def iARD_Term_matrix_form(self):
        '''the iARD term
        write the operators as matrix from, and unite all the operators to one final operator matrix,
        wicth can be act on the flattened spherical coefficients
        return: a tuple, 
            the first value is the operator matrix represent the iARD term: div.(I-pp).dr.(I-pp).grad
                with the size of [(degree+1)*(2*degree+1), (degree+1)*(2*degree+1)]
            the second is the (I-pp).dr.(I-pp).grad term which is used for calculate single fiber's orbit
        ref:[1] Huan-Chang Tseng. et. al. An objective tensor to predict anisotropic fiber orientation in 
            concentrated suspensions. Journal of Rheology. 2016(60),215.'''
        shear_rate=self.shear_rate
        CM=self.CM
        CI=self.CI
        deltax=self.deltax_matrix
        deltay=self.deltay_matrix
        deltaz=self.deltaz_matrix
        I_uu_op=self.I_uu_dot_matrix_op
        surf_deltax,surf_deltay,surf_deltaz=I_uu_op([deltax,deltay,deltaz])
        '''calculate the iARD term:
        Dr=[[CI-CI*CM,         0,          0],
              [0,                CI-CI*CM,   0],
              [0,                     0,          CI]]
        Grad=[∇_xf, ∇_yf, ∇_zf]
        (I-uu)•Dr•(I-uu)•∇_s(f)=(I-uu)•Dr•∇_s(f)=(I-uu)•(Dr•∇_s(f))
        Dr•∇_s(f)=
        [(CI-CI*CM)*∇_xf,(CI-CI*CM)*∇_yf,CI*∇_zf]'''
        dr_dot_surfgrad=[(CI-CI*CM)*deltax,(CI-CI*CM)*deltay,CI*deltaz]
        
        
        Iuu_dr_dot_surfgrad=I_uu_op(dr_dot_surfgrad)
        
        minus_Iuu_dr_dot_surfgrad=[-np.abs(shear_rate)*Iuu_dr_dot_surfgrad[0], \
                                   -np.abs(shear_rate)*Iuu_dr_dot_surfgrad[1], \
                                   -np.abs(shear_rate)*Iuu_dr_dot_surfgrad[2]]
        
        surf_div_iARD_matrix=-(surf_deltax.dot(minus_Iuu_dr_dot_surfgrad[0])+ \
                   surf_deltay.dot(minus_Iuu_dr_dot_surfgrad[1])+ \
                   surf_deltaz.dot(minus_Iuu_dr_dot_surfgrad[2]))
        return (self.revmove_zeros(surf_div_iARD_matrix),minus_Iuu_dr_dot_surfgrad)
    
    def RSC_Term_matrix_form(self,t,flatten_clm_co):
        '''the RSC term, but: the function RSC_Term_A2_Form will be faster than this one!
        note that: 
                before using this function, the function RSC_Term_matrix_form_prepare should be called first!
        
        return a tuple:
            the first is the div(q) term, the flatten type of the spherical harmonics coefficients after acted by the RSC term.
            with the size of : [(degree+1)*(2*degree+1),]
            the second is the gradient which is used for calculate single fiber's orbit'''
        shear_rate=self.shear_rate
        D=np.array([[0,shear_rate/2,0],[shear_rate/2,0,0],[0,0,0]])
        k=self.k
        deltax=self.deltax_matrix
        deltay=self.deltay_matrix
        deltaz=self.deltaz_matrix
        I_uu_op=self.I_uu_dot_matrix_op
        surf_deltax,surf_deltay,surf_deltaz=I_uu_op([deltax,deltay,deltaz])        
        clm_co=np.reshape(flatten_clm_co,[self.degree+1,2*self.degree+1])
        A2=self.get_Orientation_Tensor_A2(clm_co)
        A4=self.get_Orientation_Tensor_A4_fast(clm_co)
        #calculate the A2 A4 firstly
        lams,Q=np.linalg.eigh(A2)
        '''calculate the RSC term:
        q=beta(ee.p - ee:ppp)
        beta=....'''        
        x=self.x_flatten
        y=self.y_flatten
        z=self.z_flatten
        x3=self.x3_flatten
        y3=self.y3_flatten
        z3=self.z3_flatten
        x2y=self.x2y_flatten
        z2y=self.z2y_flatten
        x2z=self.x2z_flatten
        y2z=self.y2z_flatten
        y2x=self.y2x_flatten
        z2x=self.z2x_flatten
        xyz=self.xyz_flatten
        beta_i=[]
        for i in range(3):
            lam_i=lams[i]
            e_i=Q[:,i]    
            beta=-5*(1-k)/4/np.pi*(self.xi*(lam_i*np.dot(np.dot(D,e_i),e_i)-  \
                                     np.tensordot(np.dot(e_i,np.dot(e_i,A4)),D,axes=[[0,1],[1,0]]))+ \
                        self.CI*np.abs(shear_rate)*(1-3*lam_i))
            beta_i.append(beta)
        b1=beta_i[0]
        b2=beta_i[1]
        b3=beta_i[2]
        e11=Q[0,0]
        e21=Q[1,0]
        e31=Q[2,0]
        e12=Q[0,1]
        e22=Q[1,1]
        e32=Q[2,1]
        e13=Q[0,2]
        e23=Q[1,2]
        e33=Q[2,2]
        
        q1=(-b1*e11**2-b2*e12**2-b3*e13**2)*x3  +  (-2*b1*e11*e21-2*b2*e12*e22-2*b3*e13*e23)*x2y  +  \
            (-2*b1*e11*e31-2*b2*e12*e32-2*b3*e13*e33)*x2z  +  (-b1*e21**2-b2*e22**2-b3*e23**2)*y2x  +  \
            (-b1*e31**2-b2*e32**2-b3*e33**2)*z2x  +  (-2*b1*e21*e31-2*b2*e22*e32-2*b3*e23*e33)*xyz  +  \
            (b1*e11**2+b2*e12**2+b3*e13**2)*x  +  (b1*e11*e21+b2*e12*e22+b3*e13*e23)*y  +  \
            (b1*e11*e31+b2*e12*e32+b3*e13*e33)*z
        q2=(-b1*e21**2-b2*e22**2-b3*e23**2)*y3  +  (-b1*e11**2-b2*e12**2-b3*e13**2)*x2y  +  \
            (-b1*e31**2-b2*e32**2-b3*e33**2)*z2y  +  (-2*b1*e21*e31-2*b2*e22*e32-2*b3*e23*e33)*y2z  +  \
            (-2*b1*e11*e21-2*b2*e12*e22-2*b3*e13*e23)*y2x  +  (-2*b1*e11*e31-2*b2*e12*e32-2*b3*e13*e33)*xyz  +  \
            (b1*e11*e21+b2*e12*e22+b3*e13*e23)*x  +  (b1*e21**2+b2*e22**2+b3*e23**2)*y  +  \
            (b1*e21*e31+b2*e22*e32+b3*e23*e33)*z
        q3=(-b1*e31**2-b2*e32**2-b3*e33**2)*z3  +  (-2*b1*e21*e31-2*b2*e22*e32-2*b3*e23*e33)*z2y  +  \
            (-b1*e11**2-b2*e12**2-b3*e13**2)*x2z  +  (-b1*e21**2-b2*e22**2-b3*e23**2)*y2z  +  \
            (-2*b1*e11*e31-2*b2*e12*e32-2*b3*e13*e33)*z2x  +  (-2*b1*e11*e21-2*b2*e12*e22-2*b3*e13*e23)*xyz  +  \
            (b1*e11*e31+b2*e12*e32+b3*e13*e33)*x  +  (b1*e21*e31+b2*e22*e32+b3*e23*e33)*y  +  \
            (b1*e31**2+b2*e32**2+b3*e33**2)*z
        RSC_surf_div=-(surf_deltax.dot(q1)+surf_deltay.dot(q2)+surf_deltaz.dot(q3))
        return (RSC_surf_div,[q1,q2,q3])

    def RSC_Term_matrix_form_prepare(self):
        '''call before calling RSC_Term_matrix_form
        used for calculate the x^3, y^3, z^3 ... terms prepared for the RSC'''
        Identity=np.zeros([self.degree+1,2*self.degree+1])
        Identity[0,self.degree]=1
        Identity_flatten=Identity.flatten()
        x=self.x_matrix
        y=self.y_matrix
        z=self.z_matrix
        self.x_flatten=x.dot(Identity_flatten)
        self.y_flatten=y.dot(Identity_flatten)
        self.z_flatten=z.dot(Identity_flatten)
        self.x3_flatten=self.revmove_zeros(x.dot(x.dot(x))).dot(Identity_flatten)
        self.y3_flatten=self.revmove_zeros(y.dot(y.dot(y))).dot(Identity_flatten)
        self.z3_flatten=self.revmove_zeros(z.dot(z.dot(z))).dot(Identity_flatten)
        self.x2y_flatten=self.revmove_zeros(x.dot(x.dot(y))).dot(Identity_flatten)
        self.x2z_flatten=self.revmove_zeros(x.dot(x.dot(z))).dot(Identity_flatten)
        self.y2z_flatten=self.revmove_zeros(y.dot(y.dot(z))).dot(Identity_flatten)
        self.y2x_flatten=self.revmove_zeros(x.dot(y.dot(y))).dot(Identity_flatten)
        self.z2x_flatten=self.revmove_zeros(z.dot(z.dot(x))).dot(Identity_flatten)
        self.z2y_flatten=self.revmove_zeros(z.dot(z.dot(y))).dot(Identity_flatten)
        self.xyz_flatten=self.revmove_zeros(x.dot(y.dot(z))).dot(Identity_flatten)
    
    def RSC_Term_A2_Form(self,t,flatten_clm_co):
        '''the RSC term
        RSC term means that: subtraction the one part from A2.
        fortunately, we can transform A2 to spherical harmonics coefficients,
        so, the dA2/dt term can also be transformed to clm_co type
        
        -div(q)=-(1-k)sum(dot_Υ_i e_i e_i)
        usage:
            input: t, unused
            flatten_clm_co: the flatten type of the spherical harmonics coefficients matrix
        return:
            the div(q) term, the flatten type of the spherical harmonics coefficients after acted by the RSC term.
            with the size of : [(degree+1)*(2*degree+1),]
        ref:[1] Jin Wang. et. al. An objective model for slow orientation kinetics in concentrated fiber suspensions:
            Theory and rheological evidence. Journal of Rheology. 2008(52),1179.
        '''
        shear_rate=self.shear_rate
        D=np.array([[0,shear_rate/2,0],[shear_rate/2,0,0],[0,0,0]])
        k=self.k
        clm_co=np.reshape(flatten_clm_co,[self.degree+1,2*self.degree+1])
        A2=self.get_Orientation_Tensor_A2(clm_co)
        A4=self.get_Orientation_Tensor_A4_fast(clm_co)
        #calculate the A2 A4 firstly
        lams,Q=np.linalg.eigh(A2)
        #delta_q=np.zeros([(self.degree+1)*(2*self.degree+1),3])
        RSC_A2=np.zeros([3,3])
        for i in range(3):
            lam_i=lams[i]
            e_i=Q[:,i]
            
            dot_lam_i=2*self.xi*(lam_i*np.dot(np.dot(D,e_i),e_i)-  \
                               np.tensordot(np.dot(e_i,np.dot(e_i,A4)),D,axes=[[0,1],[1,0]]))+ \
                        2*self.CI*np.abs(shear_rate)*(1-3*lam_i)
            RSC_A2+=-(1-k)*dot_lam_i*np.tensordot(e_i,e_i,axes=0)
        RSC_clm_co=self.A2_to_clm_co(RSC_A2)
        return RSC_clm_co.flatten()
    def PRP_Term_A2_form(self,t,flatten_clm_co):
        '''the RPR term, 
        RPR term is identity to RSC term when beta=0;
        usage:
            input: t, unused
            flatten_clm_co: the flatten type of the spherical harmonics coefficients matrix
        return:
            the div(q) term, the oparator matrix of the spherical harmonics coefficients represent the RSC term.
        '''
        return self.RSC_Term_A2_Form(t,flatten_clm_co)
        
    def Confined_Term_matrix_form(self,eps,alpha):
        '''My Confined term,
        D_star=ε[[0.5+alpha,0,0],[0,-1,0],[0,0,0.5-alpha]]
        Confined_term=-∇∙[ξ(D_star∙p-D_star:ppp)ψ]
        usage:
            input:eps, the compress rate, alpha, the finetune parameter
        return:
            the div confined term, the operator matrix of the spherical harmonics coefficients represent the confined term
        '''
        shear_rate=self.shear_rate
        x=self.x_matrix
        y=self.y_matrix
        z=self.z_matrix
        xi=self.xi
        deltax=self.deltax_matrix
        deltay=self.deltay_matrix
        deltaz=self.deltaz_matrix
        I_uu_op=self.I_uu_dot_matrix_op
        surf_deltax,surf_deltay,surf_deltaz=I_uu_op([deltax,deltay,deltaz])
        # calculate the ξ(D_star∙p-D_star:ppp) ψ term
        
        confined_psi=[0.5*eps*x + alpha*eps*x - 0.5*eps*x.dot(x.dot(x)) - alpha*eps*x.dot(x.dot(x)) + \
               eps*x.dot(y.dot(y)) - 0.5*eps*x.dot(z.dot(z)) + alpha*eps*x.dot(z.dot(z)), \
               -eps*y - 0.5*eps*x.dot(x).dot(y) - alpha*eps*x.dot(x).dot(y)+eps*y.dot(y.dot(y)) - \
               0.5*eps*y.dot(z.dot(z)) + alpha*eps*y.dot(z.dot(z)),  \
               0.5*eps*z - alpha*eps*z - 0.5*eps*x.dot(x)*z - alpha*eps*x.dot(x).dot(z) + \
               eps*y.dot(y).dot(z) - 0.5*eps*z.dot(z.dot(z)) + alpha*eps*z.dot(z.dot(z))]
        surf_div_confined_matrix=-(surf_deltax.dot(confined_psi[0])+surf_deltay.dot(confined_psi[1])+surf_deltaz.dot(confined_psi[2]))
        return surf_div_confined_matrix
    
    def Contact_Term_matrix_form(self,y_L_ratio):
        '''Contact term from paper,
        dotp_C=-(dotp_HD.n)/(1-4*y_L_ratio^2)*(n-n.pp)
        usage:
            input:eps, the compress rate, alpha, the finetune parameter
        return:
            the div confined term, the operator matrix of the spherical harmonics coefficients represent the confined term
            
        ref: Perez, M. et. al. Journal of Non-Newtonian Fluid Mechanics 233, 61-74 (2016).
        '''
        shear_rate=self.shear_rate
        x=self.x_matrix
        y=self.y_matrix
        z=self.z_matrix
        xi=self.xi
        deltax=self.deltax_matrix
        deltay=self.deltay_matrix
        deltaz=self.deltaz_matrix
        I_uu_op=self.I_uu_dot_matrix_op
        surf_deltax,surf_deltay,surf_deltaz=I_uu_op([deltax,deltay,deltaz])
        # calculate the dotp ψ term
        
        
        dotp_HD=[(1+xi)*shear_rate/2*y-xi*shear_rate*x.dot(x.dot(y)), \
                (xi-1)*shear_rate/2*x-xi*shear_rate*x.dot(y.dot(y)),  \
                -xi*shear_rate*x.dot(y.dot(z))]
        dotp_HD_n_co=-dotp_HD[1]/(1-4*y_L_ratio**2)
        
        dotp_C=[-dotp_HD_n_co.dot(y.dot(x)), \
                dotp_HD_n_co-dotp_HD_n_co.dot(y.dot(y)), \
                -dotp_HD_n_co.dot(y.dot(z))]
        #dotp_C=I_uu_op(dotp_C)
        surf_div_HD_matrix=-(surf_deltax.dot(dotp_C[0])+surf_deltay.dot(dotp_C[1])+surf_deltaz.dot(dotp_C[2]))

        return (self.revmove_zeros(surf_div_HD_matrix),dotp_C)
        
    
    def run_matrix_type(self,dt=0.001,y_L_ratio=0.5):
        '''
        y_L_ratio is the paramter describe the contact term;
        see the function self.Contact_Term_matrix_form(y_L_ratio)
        '''
        psi_co_flatten=self.init_sphharm_co.flatten()
        div_matrix=self.HD_Term_matrix_form()[0]+self.iARD_Term_matrix_form()[0]
        if y_L_ratio<0.5:
            # the contact term
            div_matrix+=self.Contact_Term_matrix_form(y_L_ratio=y_L_ratio)[0]
        if self.k!=1:
            #model is the 'RSC' model, so prepare for the RSC:
            #print('RSC model is chosen, the k is: '+str(self.k))
            self.RSC_Term_matrix_form_prepare()
        res_t=[]
        res_y=[]
        A2=[]
        RSC=[]
        t_max=np.max(self.t_span)
        num=len(self.t_span)
        interv=int(t_max/dt/num)
        t_steps=np.arange(0,t_max,dt)
        for n_i,t in enumerate(t_steps):
            if self.k!=1:
                #model=='RSC':
                #RSC_term=self.RSC_Term_matrix_form(t,psi_co_flatten)[0]
                RSC_term=self.RSC_Term_A2_Form(t,psi_co_flatten)
                dpsi=div_matrix.dot(psi_co_flatten)+RSC_term
            else:
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
             'k': self.k, \
             'rho': self.rho, \
             'degree': self.degree, \
             'shear_rate': self.shear_rate, \
             'init_sphharm_co':self.init_sphharm_co, \
             'RSC': np.array(RSC)}
        return res
    
    
    def run_confined_matrix_type(self,dt=0.001,epsilon=0.04,alpha=0.5):
        '''
        The confined term is the virtal compression term. 
        '''
        self.confined_epsilon=epsilon
        self.confined_alpha=alpha
        psi_co_flatten=self.init_sphharm_co.flatten()
        div_matrix=self.HD_Term_matrix_form()[0]+self.iARD_Term_matrix_form()[0]+ \
                    self.Confined_Term_matrix_form(epsilon,alpha)
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
             'eps': epsilon, \
             'alpha': alpha, \
             'rho': self.rho, \
             'degree': self.degree, \
             'shear_rate': self.shear_rate, \
             'init_sphharm_co':self.init_sphharm_co}
        return res
    def cal_single_fiber_orbit_confined(self,p0,psi_clm,max_time=100,num=1000,max_step=0.05):
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
            D=0.5*(L.T+L)+self.confined_epsilon*(np.array([[0.5+self.confined_alpha,0,0],[0,1,0],[0,0,0.5-self.confined_alpha]]))
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
    def cal_single_fiber_orbit_matrix_Form(self,p0,psi_clm,max_time=100,num=100,max_step=0.05):
        '''calculate the trajetory of single fiber predicted by the model.
            different from the function cal_single_fiber_orbit, this function use the spherical harmonics 
            method to calculate the q and CI dr dot_p_HD term inside the function
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
        Identity_clm_co=np.zeros([self.degree+1,2*self.degree+1])
        Identity_clm_co[0,self.degree]=1
        Identity_flatten_clm_co=Identity_clm_co.flatten()
        
        psi_flatten_clm_co=self.to_clm_co(psi_clm).flatten()
        #dot_p HD term
        dot_p_HD_clm_x=self.to_clm(np.reshape(self.HD_Term_matrix_form()[1][0].dot(Identity_flatten_clm_co),[self.degree+1,2*self.degree+1]))
        dot_p_HD_clm_y=self.to_clm(np.reshape(self.HD_Term_matrix_form()[1][1].dot(Identity_flatten_clm_co),[self.degree+1,2*self.degree+1]))
        dot_p_HD_clm_z=self.to_clm(np.reshape(self.HD_Term_matrix_form()[1][2].dot(Identity_flatten_clm_co),[self.degree+1,2*self.degree+1]))
        # calculate the ln(psi)
        psi_grid=psi_clm.expand(grid='DH')
        ln_psi_grid=psi_grid.copy()
        ln_psi_grid.data=np.log(np.real(psi_grid.data))
        ln_psi_flatten_clm_co=self.to_clm_co(ln_psi_grid.expand()).flatten()
        #iARD term
        #negetive of the results,due to the transform between theta,phi and lon lats exist a negetive
        minus_Iuu_dr_dot_surfgrad_x_ln_psi_clm=self.to_clm(np.reshape(self.iARD_Term_matrix_form()[1][0].dot(ln_psi_flatten_clm_co),[self.degree+1,2*self.degree+1]))
        minus_Iuu_dr_dot_surfgrad_y_ln_psi_clm=self.to_clm(np.reshape(self.iARD_Term_matrix_form()[1][1].dot(ln_psi_flatten_clm_co),[self.degree+1,2*self.degree+1]))
        minus_Iuu_dr_dot_surfgrad_z_ln_psi_clm=self.to_clm(np.reshape(self.iARD_Term_matrix_form()[1][2].dot(ln_psi_flatten_clm_co),[self.degree+1,2*self.degree+1]))
        #RSC's q term
        self.RSC_Term_matrix_form_prepare()
        RSC_q_flatten_clm_co=self.RSC_Term_matrix_form(0,psi_flatten_clm_co)[1]
        RSC_q_clm_x=self.to_clm(np.reshape(RSC_q_flatten_clm_co[0],[self.degree+1,2*self.degree+1]))
        RSC_q_clm_y=self.to_clm(np.reshape(RSC_q_flatten_clm_co[1],[self.degree+1,2*self.degree+1]))
        RSC_q_clm_z=self.to_clm(np.reshape(RSC_q_flatten_clm_co[2],[self.degree+1,2*self.degree+1]))
        def fun_for_ode(t,p):
            #the pyshtools using the lons and lats, we should transform the phi theta to lats and lons
            p=p/np.linalg.norm(p,ord=2)
            #calculate the theta phi and lats, lons
            x=p[0]
            y=np.array(p[1])
            z=np.array(p[2])
            theta=np.arccos(z)
            if x>=0:
                phi=np.arctan(y/x)
            else:
                phi=(np.arctan(y/x)+np.pi)
            lat=float(-(theta-np.pi/2)/np.pi*180)
            lon=float(np.mod(-(phi-np.pi)/np.pi*180,360))
            # calulate the psi at the point of [theta,phi]
            psi_point=np.real(psi_clm.expand(grid='DH',lat=lat,lon=lon))
            dx=np.real(dot_p_HD_clm_x.expand(grid='DH',lat=lat,lon=lon)+ 
                       minus_Iuu_dr_dot_surfgrad_x_ln_psi_clm.expand(grid='DH',lat=lat,lon=lon)+ \
                       RSC_q_clm_x.expand(grid='DH',lat=lat,lon=lon)/psi_point)
            dy=np.real(dot_p_HD_clm_y.expand(grid='DH',lat=lat,lon=lon)+ 
                       minus_Iuu_dr_dot_surfgrad_y_ln_psi_clm.expand(grid='DH',lat=lat,lon=lon)+ \
                       RSC_q_clm_y.expand(grid='DH',lat=lat,lon=lon)/psi_point)
            dz=np.real(dot_p_HD_clm_z.expand(grid='DH',lat=lat,lon=lon)+ 
                       minus_Iuu_dr_dot_surfgrad_z_ln_psi_clm.expand(grid='DH',lat=lat,lon=lon)+ \
                       RSC_q_clm_z.expand(grid='DH',lat=lat,lon=lon)/psi_point)
            return np.array([dx,dy,dz])
        #perform the RK45 method to caulculate the evolution of p
        t_span=[0,max_time*1.01]
        t_eval=np.linspace(0,max_time,num)
        res=integrate.solve_ivp(fun_for_ode,t_span,p0,t_eval=t_eval,method='RK45',rtol=1e-4,atol=1e-7,max_step=max_step)
        df=pd.DataFrame(res.y.T,index=res.t,columns=['x','y','z'])
        df.loc[:,'theta']=np.arccos(df['z'])
        df.loc[:,'phi']=np.arctan(df['y']/df['x'])
        return df        

    def update_res_for_fiber_orbits_2(self,res_of_run_matrix_type):
        '''update the res for calculate the fiber orbits of the functin self.cal_single_fiber_orbit_2()
        usage:
        input:res_of_run_matrix_type: the results of the function self.run_matrix_type, which contains the series of 
                                the spherical harmonics coefficient object (Coeffi) provided by pyshtools packages;
        
        return: res_for_fiber_orbits_2 used in function self.cal_single_fiber_orbit_2()
        '''
        #calculate the spherica harmonics coefficient of the psi
        psi_clm_list=[]
        nabla_ln_psi_clm_x_list=[]
        nabla_ln_psi_clm_y_list=[]
        nabla_ln_psi_clm_z_list=[]
        A2_list=[]
        A4_list=[]
        for co_ind,co_i in enumerate(res_of_run_matrix_type['time_span']):
            psi_clm=self.to_clm(res_of_run_matrix_type['sphharm_co'][co_ind,:,:])
            psi_clm_list.append(psi_clm)
            psi_grid=psi_clm.expand(grid='DH')
            psi_grid.data[psi_grid.data<1E-7]=1E-7
            A2=self.get_Orientation_Tensor_A2(self.to_clm_co(psi_clm))
            A2_list.append(A2)
            A4=self.get_Orientation_Tensor_A4(self.to_clm_co(psi_clm))
            A4_list.append(A4)
            # calculate the ln(psi)
            ln_psi_grid=psi_grid.copy()
            ln_psi_grid.data=np.log(np.real(psi_grid.data))
            
            ln_psi_clm=ln_psi_grid.expand()
            #calculate the \nabla ln(psi)
            nabla_ln_psi_clm_x=self.to_clm(self.deltax(self.to_clm_co(ln_psi_clm)))
            nabla_ln_psi_clm_y=self.to_clm(self.deltay(self.to_clm_co(ln_psi_clm)))
            nabla_ln_psi_clm_z=self.to_clm(self.deltaz(self.to_clm_co(ln_psi_clm)))
            nabla_ln_psi_clm_x_list.append(nabla_ln_psi_clm_x)
            nabla_ln_psi_clm_y_list.append(nabla_ln_psi_clm_y)
            nabla_ln_psi_clm_z_list.append(nabla_ln_psi_clm_z)
        res_for_fiber_orbits_2=res_of_run_matrix_type.copy()
        res_for_fiber_orbits_2['nabla_ln_psi_clm_x_list']=nabla_ln_psi_clm_x_list
        res_for_fiber_orbits_2['nabla_ln_psi_clm_y_list']=nabla_ln_psi_clm_y_list
        res_for_fiber_orbits_2['nabla_ln_psi_clm_z_list']=nabla_ln_psi_clm_z_list
        res_for_fiber_orbits_2['A2_list']=A2_list
        res_for_fiber_orbits_2['A4_list']=A4_list
        res_for_fiber_orbits_2['psi_clm_list']=psi_clm_list
        return res_for_fiber_orbits_2



    def cal_single_fiber_orbit_2(self,p0,res_for_fiber_orbits_2,max_time=100,num=1000,max_step=0.2):
        '''calculate the trajetory of single fiber predicted by the model, the orientation distribution is changing with time.
        first call the self.update_res_for_fiber_orbits_2()
        usage:
        input: p0:the unit orientation vector at time=0;
                res_for_fiber_orbits_2: the results of the function self.update_res_for_fiber_orbits_2(), 
                        which contains the series of the spherical harmonics coefficient object (Coeffi) provided by pyshtools packages;
                max_time: I always set the shear rate ==0 , so the max time means the max strain to be calculated
                num: the sampling number of the trajetory of single fiber
                max_step: the maximum step during calculate the tracjetory used in RK45 method
        
        return: a DataFrame object contains trajetory of single fiber
                the index is the time; and the columns is ['x','y','z','theta','phi']
        note in this function:
            the + - of the W.p term is very important, that it will influence the period behavior!
            just becareful if should modify this function again!
            if this was wrong, it will not rotate numericallly!'''
        # def the function for ode to calculate the unit vector
        def fun_for_ode(t,p):
            '''intput:
                t: the strain of fiber
                p: the orientation vector of the fiber
                return: the change of the p'''
            ###########################################################################
            #extract the nabla_ln_psi_clm_x nabla_ln_psi_clm_y nabla_ln_psi_clm_z A2 A4
            #import ipdb
            #ipdb.set_trace()
            time_span=res_for_fiber_orbits_2['time_span']
            co_ind=np.where(time_span>=t)[0]
            if len(co_ind)==0:
                co_ind=-1
            else:
                co_ind=co_ind[0]
            psi_clm=res_for_fiber_orbits_2['psi_clm_list'][co_ind]
            A2=res_for_fiber_orbits_2['A2_list'][co_ind]
            A4=res_for_fiber_orbits_2['A4_list'][co_ind]
            nabla_ln_psi_clm_x=res_for_fiber_orbits_2['nabla_ln_psi_clm_x_list'][co_ind]
            nabla_ln_psi_clm_y=res_for_fiber_orbits_2['nabla_ln_psi_clm_y_list'][co_ind]
            nabla_ln_psi_clm_z=res_for_fiber_orbits_2['nabla_ln_psi_clm_z_list'][co_ind]
            ############################################################################
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
            D=0.5*(L.T+L)
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
            #calculate the RSC term
            #only calculate the RSC term when k!=1
            if self.k!=1:
                lam,Q=np.linalg.eigh(A2)
                #closure approximation is a very bad idea
                #A4=models.a4_IBOF.a4_IBOF(A2)
                k=self.k
                q=np.zeros(3)
                for i in range(3):
                    lam_i=lam[i]
                    e_i=Q[:,i]
                    beta_i=-5*(1-k)/4/np.pi*(self.xi*(lam_i*np.dot(np.dot(D,e_i),e_i)-  \
                                np.tensordot(np.dot(e_i,np.dot(e_i,A4)),D,axes=[[0,1],[1,0]]))+ \
                                self.CI*np.abs(shear_rate)*(1-3*lam_i))
                    q+=beta_i*(np.dot(e_i,p)*e_i-np.dot(e_i,p)**2*p)
            else:
                q=np.zeros(3)
            # calulate the psi at the point of [theta,phi]
            psi_point=np.real(psi_clm.expand(grid='DH',lat=lat,lon=lon))
            return dot_p_HD-Ipp_dr_Ipp_nabla_ln_psi+q/psi_point
        #perform the RK45 method to caulculate the evolution of p
        t_span=[0,max_time*1.01]
        t_eval=np.linspace(0,max_time,num)
        res=integrate.solve_ivp(fun_for_ode,t_span,p0,t_eval=t_eval,method='RK45',rtol=1e-3,atol=1e-6,max_step=max_step)
        df=pd.DataFrame(res.y.T,index=res.t,columns=['x','y','z'])
        df.loc[:,'theta']=np.arccos(df['z'])
        df.loc[:,'phi']=np.arctan(df['y']/df['x'])
        return df
    
    
    def cal_single_fiber_orbit(self,p0,psi_clm,max_time=100,num=1000,max_step=0.05):
        '''this is used to calculate the orbit of the fibers by changing the psi_clm along with the strain
        calculate the trajetory of single fiber predicted by the model.
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
        A2=self.get_Orientation_Tensor_A2(self.to_clm_co(psi_clm))
        A4=self.get_Orientation_Tensor_A4(self.to_clm_co(psi_clm))
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
            D=0.5*(L.T+L)
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
            #calculate the RSC term
            #only calculate the RSC term when k!=1
            if self.k!=1:
                lam,Q=np.linalg.eigh(A2)
                #closure approximation is a very bad idea
                #A4=models.a4_IBOF.a4_IBOF(A2)
                k=self.k
                q=np.zeros(3)
                for i in range(3):
                    lam_i=lam[i]
                    e_i=Q[:,i]
                    beta_i=-5*(1-k)/4/np.pi*(self.xi*(lam_i*np.dot(np.dot(D,e_i),e_i)-  \
                                np.tensordot(np.dot(e_i,np.dot(e_i,A4)),D,axes=[[0,1],[1,0]]))+ \
                                self.CI*np.abs(shear_rate)*(1-3*lam_i))
                    q+=beta_i*(np.dot(e_i,p)*e_i-np.dot(e_i,p)**2*p)
            else:
                q=np.zeros(3)
            # calulate the psi at the point of [theta,phi]
            psi_point=np.real(psi_clm.expand(grid='DH',lat=lat,lon=lon))
            return dot_p_HD-Ipp_dr_Ipp_nabla_ln_psi+q/psi_point
        #perform the RK45 method to caulculate the evolution of p
        t_span=[0,max_time*1.01]
        t_eval=np.linspace(0,max_time,num)
        res=integrate.solve_ivp(fun_for_ode,t_span,p0,t_eval=t_eval,method='RK45',rtol=1e-4,atol=1e-7,max_step=max_step)
        df=pd.DataFrame(res.y.T,index=res.t,columns=['x','y','z'])
        df.loc[:,'theta']=np.arccos(df['z'])
        df.loc[:,'phi']=np.arctan(df['y']/df['x'])
        return df
    
    
    #this is the function which calculate the map of the period strain by given spherical harmonic coefficients
    def cal_period_strain_map(self,psi_clm,max_strain=400,theta_num=20):
        ''' change the initial angle of the fiber, calculate the trajetory of the fiber. And return the period strain map.
        uasage: 
        input:    psi_clm: Spherical Harmonics coeefficients object of the psi
                    theta_num, the number-1 of the thetas to be calcualted
        return:   tupule with:
                    0: pandas DataFrame，the rotation period strain of single fiber with the initial theta phi
                    1: flattened rotation period strain for 3D plotting
                    2: average of the period strain by guasian Legendre quadrature mehod'''
        import timeit
        from scipy import interpolate
        #the period strain will be stored in the period_df DataFrame 
        period_df=pd.DataFrame()
        period_df.index.name='theta'
        period_df.columns.name='phi'
        from scipy.signal import find_peaks
        #select the intial theta and phi to calculate the period strain
        psi_grid=psi_clm.expand(grid='GLQ',lmax=theta_num)
        thetas=-psi_grid.lats()/180*np.pi+np.pi/2
        phis=-psi_grid.lons()/180*np.pi+np.pi
        # calculate period strain in different theta and phi
        for theta in thetas[:int(np.floor(theta_num/2+1))]:
            t0=timeit.time.time()
            for phi in phis:
                p0=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
                df=self.cal_single_fiber_orbit(p0,psi_clm,max_strain,num=max_strain*20,max_step=0.5)
                p,_=find_peaks(np.abs(df.loc[:,'phi']))
                #calculate the period strain
                period_df.loc[theta,phi]=2*(p[1]-p[0])/20
            t1=timeit.time.time()
            print(theta,period_df.loc[theta,phi],'--time->',t1-t0)
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
        #-------------------------------------------------------------------------
        #calculate the average of the period strain of single fiber using Guassian Legendre quadrature
        #-------------------------------------------------------------------------
        psi_grid_GLQ=psi_clm.expand(grid='GLQ',lmax=20)
        
        lons_mesh_GLQ,lats_mesh_GLQ=np.meshgrid(psi_grid_GLQ.lons(),psi_grid_GLQ.lats())
        psi_data_GLQ=psi_clm.expand(lat=list(lats_mesh_GLQ.flatten()),lon=list(lons_mesh_GLQ.flatten()))
        psi_data_GLQ=np.reshape(np.array(psi_data_GLQ),(len(psi_grid_GLQ.lats()),-1)).real
        
        #calculate the avrage of the period strain
        df_period_psi=period_df.copy()*psi_data_GLQ
        Legendre_quadrature_weights=pd.DataFrame(np.tile(psi_grid.weights,[len(phis),1]).T)
        df_period_psi*=Legendre_quadrature_weights.values
        #the Legendre quadrature is valid in theta direction, but in phi direction, should multiply by np.pi*2/len(phis)
        period_average=df_period_psi.sum().sum()*np.pi*2/len(phis)

        return (period_df,period_df_flatten,period_average)
    
    def  fit_Jeffery(self,df_fit):
        '''
        df_fit is the return of the function self.cal_single_fiber_orbit()
        df_fit=a.cal_single_fiber_orbit(p0=[-1,0,0],psi_clm=psi_clm)
        df_fit.loc[df_fit.loc[:,'phi']<0,'phi']+=np.pi
        #find the several sections of the periods
        df_inital_points=df_fit.loc[df_fit.loc[:,'phi'].diff().abs()>1,:]
        df_fit.loc[:df_inital_points.index[0],'phi']=np.pi
        df_fit.loc[df_inital_points.index[1]:,'phi']=0
        
        return the effective shear rate
        '''
        #Jeffery fun for ode
        def Jeffery_fun(t,p):
            '''t is time, p is the unit orientation tensor'''
            L=np.zeros((3,3),dtype=np.float64)
            L[0,1]=1
            I=np.eye(3)
            D=0.5*(L.T+L)
            W=0.5*(L-L.T)
            return np.dot(W,p)+0.99*(np.dot(D,p)-np.dot(np.dot(D,p),p)*p)
        def cal_Jeffry_orbit():
            '''effect_shear_rate is the fitting parameters
            strain_shift is the strain shift
            '''
            #fit process by Jeffery
            t_span=(0,400)
            p0=np.array([-1,0,0])
            res_jeffery=scipy.integrate.solve_ivp(Jeffery_fun,t_span=t_span,y0=p0,t_eval=df_fit.index)
            df_fit.loc[:,'x_Jeffery']=res_jeffery.y[0,:]
            df_fit.loc[:,'y_Jeffery']=res_jeffery.y[1,:]
            df_fit.loc[:,'z_Jeffery']=res_jeffery.y[2,:]
            df_fit.loc[:,'theta_Jeffery']=np.arccos(df_fit.loc[:,'z_Jeffery'])
            df_fit.loc[:,'phi_Jeffery']=np.arctan(df_fit.loc[:,'y_Jeffery']/df_fit.loc[:,'x_Jeffery'])
            df_fit.loc[df_fit.loc[:,'phi_Jeffery']<0,'phi_Jeffery']+=np.pi
            #find the several sections of the periods
            df_inital_points_Jeffery=df_fit.loc[df_fit.loc[:,'phi_Jeffery'].diff().abs()>1,:]
            df_fit.loc[:df_inital_points_Jeffery.index[0],'phi_Jeffery']=np.pi
            df_fit.loc[df_inital_points_Jeffery.index[1]:,'phi_Jeffery']=0
        def fun_for_minimize(x):
            '''x=np.array([effect_shear_rate, strain_shift])
            '''
            effect_shear_rate=x[0]
            strain_shift=x[1]
            #calculate the shifted values
            # fill the value
            x_interp=df_fit.index/effect_shear_rate+strain_shift
            x_interp=x_interp.values
            x_interp[0]=-1000
            x_interp[-1]=1000
            #calculate the interpolate values
            f_interp=scipy.interpolate.interp1d(x_interp,df_fit.loc[:,'phi_Jeffery'])
            df_fit.loc[:,'phi_Jeffery_new']=f_interp(df_fit.index)
            df_fit.loc[:,'err']=np.abs(df_fit.loc[:,'phi_Jeffery_new']-df_fit.loc[:,'phi'])
            return df_fit.loc[:,'err'].sum()
        cal_Jeffry_orbit()
        #np.abs(df_fit.loc[:,'phi_Jeffery']-df_fit.loc[:,'phi'])
        #optimize of the function
        x0=[1,-20]
        res=scipy.optimize.minimize(fun_for_minimize,x0=x0)
        #df_fit.plot(y=['phi','phi_Jeffery_new'])
        return res.x[0]
    
        #this is the function which calculate the map of the period strain by given spherical harmonic coefficients
    def cal_effective_shear_rate_map(self,psi_clm,max_strain=400,theta_num=20):
        ''' change the initial angle of the fiber, calculate the trajetory of the fiber. And return the period strain map.
        uasage: 
        input:    psi_clm: Spherical Harmonics coeefficients object of the psi
                    theta_num, the number-1 of the thetas to be calcualted
        return:   tupule with:
                    0: pandas DataFrame，the rotation period strain of single fiber with the initial theta phi
                    1: flattened rotation period strain for 3D plotting
                    2: average of the period strain by guasian Legendre quadrature mehod'''
        import timeit
        from scipy import interpolate
        #the period strain will be stored in the effect_rate_df DataFrame 
        effect_rate_df=pd.DataFrame()
        effect_rate_df.index.name='theta'
        effect_rate_df.columns.name='phi'
        from scipy.signal import find_peaks
        #select the intial theta and phi to calculate the period strain
        psi_grid=psi_clm.expand(grid='GLQ',lmax=theta_num)
        thetas=-psi_grid.lats()/180*np.pi+np.pi/2
        phis=-psi_grid.lons()/180*np.pi+np.pi
        # calculate period strain in different theta and phi
        for theta in thetas[:int(np.floor(theta_num/2+1))]:
            t0=timeit.time.time()
            for phi in phis:
                p0=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
                df=self.cal_single_fiber_orbit(p0,psi_clm,max_strain,num=max_strain*20,max_step=0.5)
                
                #adjust the phi, 
                df.loc[df.loc[:,'phi']<0,'phi']+=np.pi
                df_fit=df.copy()
                #find the several sections of the periods
                df_inital_points=df_fit.loc[df_fit.loc[:,'phi'].diff().abs()>1,:]
                df_fit.loc[:df_inital_points.index[0],'phi']=np.pi
                df_fit.loc[df_inital_points.index[1]:,'phi']=0
                
                #calculate the effective shear rate
                effect_rate_df.loc[theta,phi]=self.fit_Jeffery(df_fit)
            t1=timeit.time.time()
            print(theta,effect_rate_df.loc[theta,phi],'--time->',t1-t0)
        # due to we only calculate half of the thetaes, so we should add the symmetric part about theta
        if theta_num%2==0:
            for theta in effect_rate_df.index[:-1]:
                effect_rate_df.loc[np.pi-theta,:]=effect_rate_df.loc[theta,:]
        else:
            for theta in effect_rate_df.index:
                effect_rate_df.loc[np.pi-theta,:]=effect_rate_df.loc[theta,:]
        effect_rate_df=effect_rate_df.sort_index()
        #flatten type of effect_rate_df, used for 3D plotting;
        #usage: from mpl_toolkits.mplot3d import axes3d
        #ax = pylab.gca(projection='3d')
        #ax.plot(effect_rate_df_flatten.theta,effect_rate_df_flatten.phi,effect_rate_df_flatten.period_strain)
        theta_span,phi_span=np.meshgrid(effect_rate_df.index,effect_rate_df.columns)
        flatten_data=np.array(effect_rate_df).flatten()
        effect_rate_df_flatten=pd.DataFrame([theta_span.T.flatten(),phi_span.T.flatten(),flatten_data], \
                           index=['theta','phi','period_strain']).T
        #-------------------------------------------------------------------------
        #calculate the average of the period strain of single fiber using Guassian Legendre quadrature
        #-------------------------------------------------------------------------
        psi_grid_GLQ=psi_clm.expand(grid='GLQ',lmax=20)
        
        lons_mesh_GLQ,lats_mesh_GLQ=np.meshgrid(psi_grid_GLQ.lons(),psi_grid_GLQ.lats())
        psi_data_GLQ=psi_clm.expand(lat=list(lats_mesh_GLQ.flatten()),lon=list(lons_mesh_GLQ.flatten()))
        psi_data_GLQ=np.reshape(np.array(psi_data_GLQ),(len(psi_grid_GLQ.lats()),-1)).real
        
        #calculate the avrage of the period strain
        df_period_psi=effect_rate_df.copy()*psi_data_GLQ
        Legendre_quadrature_weights=pd.DataFrame(np.tile(psi_grid.weights,[len(phis),1]).T)
        df_period_psi*=Legendre_quadrature_weights.values
        #the Legendre quadrature is valid in theta direction, but in phi direction, should multiply by np.pi*2/len(phis)
        effective_rate_average=df_period_psi.sum().sum()*np.pi*2/len(phis)

        return (effect_rate_df,effect_rate_df_flatten,effective_rate_average)