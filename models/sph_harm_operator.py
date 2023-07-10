import numpy as np
import pyshtools as sht
import pyshtools
import scipy 
from scipy import sparse

class sph_harm_operator:
    '''operators onto the spherical harmonics' coefficient
    
    the coeff of the shtools should be changed by:
    clm_co=np.concatenate((np.fliplr(clm2.coeffs[1,:,1:]),clm2.coeffs[0,:,:]),axis=1)

    and the coeff can be changed back by:
    clm_new=pyshtools.SHCoeffs.from_zeros(degree)
    clm_new.coeffs[0,:,:]=clm_co[:,degree:]
    clm_new.coeffs[1,:,1:]=np.fliplr(clm_co[:,:degree])
    '''
    def __init__(self,degree):
        '''initial the spherical harmonics coeffcients operators
        input: degree: the truncted degree of the l of the spherical harmonics coefficients
        
        the operators defined here can be found in the ref:
            [1] S. Montgomery-Smith. et. al. A systematic approach to obtaining numerical solutions of 
            Jeffery’s type equations using Spherical Harmonics Composites: Part A. 2010(41),pp:827–835
            [2] R. G. Larson. et. al. Effect of Molecular Elasticity on Out-of-Plane Orientations in 
            Shearing Flows of Liquid-Crystalline Polymers. Macromolecules 1991, 24, 6270-6282
            [3] R. G. Larson. Arrested Tumbling in Shearing Flows of Liquid Crystal Polymers. 
            Macromolecules 1990, 23, 3983-3992
        '''
        self.degree=degree
        # the mask used in this operation 
        clm_tmp=sht.SHCoeffs.from_zeros(degree)
        mask=~np.concatenate((np.fliplr(clm_tmp.mask[1,:,1:]),clm_tmp.mask[0,:,:]),axis=1)
        self.mask=mask
        # n,m used for pre coefficient of the spherical harmonics
        n_span=np.arange(0,degree+1,1)
        m_span=np.arange(-degree,degree+1,1)
        n,m=np.meshgrid(n_span,m_span)
        n=n.T
        m=m.T
        n[mask]=0
        m[mask]=0
        self.n=n
        self.m=m
        self.lplus_PreCo=-np.sqrt((n+m)*(n-m+1))
        self.lminus_PreCo=-np.sqrt((n-m)*(n+m+1))
        self.zminus_PreCo=np.sqrt((n+m)*(n-m)/((2*n-1)*(2*n+1)))
        self.zplus_PreCo=np.sqrt((n+m+1)*(n-m+1)/((2*n+1)*(2*n+3)))
        self.zplus_PreCo[mask]=0
        self.L2_PreCo=n*(n+1)
        self._generate_operator_matrix()
        
        #symetric mask 
        # when the solution is symetirc about x-z plane the coefficients are real
        #and also apply ψ(-p)=ψ(p) the odd l is zero, only the even l left
        #notice that the solution si symetric about x-z plane is the pre-condition
        apply_symetric_clm_co_mask=mask.copy()
        odd_degrees=np.arange(1,degree+1,2)
        apply_symetric_clm_co_mask[odd_degrees,:]=True
        self.apply_symetric_clm_co_mask=apply_symetric_clm_co_mask
        self.apply_symetric_flatten_clm_co_mask=apply_symetric_clm_co_mask.flatten()
        
    def to_clm(self,clm_co,kind='complex',normalization='4pi',csphase=1):
        '''change the clm coefficient in this class to the clm object used in pyshtools'''
        degree=clm_co.shape[0]-1
        if degree!=self.degree:
            print('warning: the degree is not consistent with the sph_harm_operator class')
        clm=sht.SHCoeffs.from_zeros(degree,kind=kind,normalization=normalization,csphase=csphase)
        clm.coeffs[0,:,:]=clm_co[:,degree:]
        clm.coeffs[1,:,1:]=np.fliplr(clm_co[:,:degree])
        return clm
    def to_clm_co(self,clm):
        '''change the clm object used in the pyshtools to the clm coefficient used in this class'''
        degree=clm.degrees()[-1]
        if degree!=self.degree:
            print('warning: the degree is not consistent with the sph_harm_operator class')
            self.__init__(degree)
        clm_co=np.concatenate((np.fliplr(clm.coeffs[1,:,1:]),clm.coeffs[0,:,:]),axis=1)
        return clm_co
    def lplus(self,clm_co):
        '''pyshtools.SHCoeffs.from_zeros(degree)
        clm_co=np.concatenate((np.fliplr(clm2.coeffs[1,:,1:]),clm2.coeffs[0,:,:]),axis=1)
        L+ operator of the coefficient of the spherical harmonic'''
        # check the degree of the spherical harmonic
        degree=clm_co.shape[0]-1
        if degree!=self.degree:
            self.__init__(degree)
        # multiply by the np.sqrt((n+m)(n-m+1))
        clm_co_with_pre=clm_co*self.lplus_PreCo
        #new coefficient object and perform the L+ operator , shift the m to m-1, 
        clm_co_new=np.zeros_like(clm_co)
        clm_co_new[:,:-1]=clm_co_with_pre[:,1:]
        return clm_co_new        
    def lminus(self,clm_co):
        '''L- operator of the coefficient of the spherical harmonic
        pyshtools.SHCoeffs.from_zeros(degree)
        clm_co=np.concatenate((np.fliplr(clm2.coeffs[1,:,1:]),clm2.coeffs[0,:,:]),axis=1)'''
        # check the degree of the spherical harmonic
        degree=clm_co.shape[0]-1
        if degree!=self.degree:
            self.__init__(degree)
        # multiply by the np.sqrt((n-m)(n+m+1))
        clm_co_with_pre=clm_co*self.lminus_PreCo
        #new coefficient object and perform the L- operator , shift the m to m+1, 
        clm_co_new=np.zeros_like(clm_co)
        clm_co_new[:,1:]=clm_co_with_pre[:,:-1]
        return clm_co_new
    def z(self, clm_co):
        '''z operator of the spherical harmonics'''
        # check the degree of the spherical harmonic
        degree=clm_co.shape[0]-1
        if degree!=self.degree:
            self.__init__(degree)
        # multiply by the pre coefficient
        mask=self.mask
        # the n-1 term
        clm_co_minus=clm_co*self.zminus_PreCo
        #new coefficient object and perform the z operator , shift the n to n+1, and n-1 
        c_minus=np.zeros_like(clm_co)
        c_minus[:-1,:]=clm_co_minus[1:,:]
        #the n+1 term 
        clm_co_plus=clm_co*self.zplus_PreCo
        c_plus=np.zeros_like(clm_co)
        c_plus[1:,:]=clm_co_plus[:-1,:]
        # the result
        clm_co_new=c_minus+c_plus
        clm_co_new[mask]=0
        return clm_co_new
    def Lz(self,clm_co):
        ''' perform the Lz operator the spherical harmonica coefficient'''
        return -self.m*clm_co
    def Lx(self,clm_co):
        '''perform the Lx operator'''
        return 0.5*(self.lminus(clm_co)+self.lplus(clm_co))
    def Ly(self,clm_co):
        '''perform the Ly operator'''
        return complex(0,-0.5)*(-self.lminus(clm_co)+self.lplus(clm_co))
    def L2(self,clm_co):
        '''perform the L^2 operator'''
        return self.L2_PreCo*clm_co
    def x(self,clm_co):
        '''perform the x operator'''
        return complex(0,1)*self.z(self.Ly(clm_co))-complex(0,1)*self.Ly(self.z(clm_co))
    def y(self,clm_co):
        '''perform the y operator'''
        return complex(0,1)*self.Lx(self.z(clm_co))-complex(0,1)*self.z(self.Lx(clm_co))
    def deltax(self,clm_co):
        '''gradient in the x axis, only for sphere'''
        return complex(0,1)*self.z(self.Ly(clm_co))-complex(0,1)*self.y(self.Lz(clm_co))
    def deltay(self,clm_co):
        '''gradient in the x axis, only for sphere'''
        return complex(0,1)*self.x(self.Lz(clm_co))-complex(0,1)*self.z(self.Lx(clm_co))
    def deltaz(self,clm_co):
        '''gradient in the x axis, only for sphere'''
        return complex(0,1)*self.y(self.Lx(clm_co))-complex(0,1)*self.x(self.Ly(clm_co))
    def divergence(self,clm_v):
        '''clm_v: a list of clm
        input clm_v is the vector on the surface of a unit sphere
        the operator is ((I-uu)∙∇)∙f=∇∙f'''
        return self.deltax(clm_v[0])+self.deltay(clm_v[1])+self.deltaz(clm_v[2])
    def divergence_Iuu_tpye(self,clm_v):
        '''
        type of clm_v: list
        input clm_v is the vector in the spherical surface
        the operator is ((I-uu)∙∇)∙f
        which is equals to (I-uu):(∇∙f)
        ((I-uu)∙∇)∙f  can be write as:
        {(1-x^2)∇_x- xy∇_y- xz∇_z;
        -xy∇_x+(1-y^2)∇_y-yz∇_z;
        -xz∇_x-yz∇_y+(1-z^2)∇_z} ∙ {f_x, f_y, f_z}
        =
        (1-x^2)∇_x(f_x)- xy∇_y(f_x)- xz∇_z(f_x)+ \
        -xy∇_x(f_y)+(1-y^2)∇_y(f_y)-yz∇_z(f_y)+ \
        -xz∇_x(f_z)-yz∇_y(f_z)+(1-z^2)∇_z(f_z)
        '''
        dxfx=self.deltax(clm_v[0])
        dxfy=self.deltax(clm_v[1])
        dxfz=self.deltax(clm_v[2])
        dyfx=self.deltay(clm_v[0])
        dyfy=self.deltay(clm_v[1])
        dyfz=self.deltay(clm_v[2])
        dzfx=self.deltaz(clm_v[0])
        dzfy=self.deltaz(clm_v[1])
        dzfz=self.deltaz(clm_v[2])
        res=dxfx-self.x(self.x(dxfx))-self.x(self.y(dyfx))-self.x(self.z(dzfx)) \
            -self.x(self.y(dxfy))+dyfy-self.y(self.y(dyfy))-self.y(self.z(dzfy)) \
            -self.x(self.z(dxfz))-self.y(self.z(dyfz))+dzfz-self.z(self.z(dzfz))
        return res
    def I_uu_dot(self,clm_co_v):
        '''
        (I-uu)•f operator
        fv is the vector of the  spherical harmonics' coefficients
        (I-uu)•f={{1-x^2,  xy,   xz};
                    {xy,  1-y^2,   yz};
                    {xz,   yz,  1-z^2}}•{f_x,f_y,f_z}
        which can be write as:
        {(1-x^2)f_x- xyf_y- xzf_z;
        -xyf_x+(1-y^2)f_y-yzf_z;
        -xzf_x-yzf_y+(1-z^2)f_z;}
        
        the above is too slow
        
        can be write as:f-u(u•f)
        {fx, fy, fz}-{x,y,z}*(x*fx+y*fy+z*fz)
        =
        {fx-(u•f)*x,
        fy-(u•f)*y,
        fz-(u•f)*z}
        '''
        fx=clm_co_v[0]
        fy=clm_co_v[1]
        fz=clm_co_v[2]
        x=self.x
        y=self.y
        z=self.z
        '''
        #abondan
        res=[fx-x(x(fx))-x(y(fy))-x(z(fz)),
              -x(y(fx))+fy-y(y(fy))-y(z(fz)),
              -x(z(fx))-y(z(fy))+fz-z(z(fz))]
        '''
        uf=x(fx)+y(fy)+z(fz)
        res=[fx-x(uf), fy-y(uf), fz-z(uf)]
        return res
    def surface_gradient(self,clm_co):
        '''compute the surface gradient by (I-uu)•∇f operator;
        which is equal to (-i u × L)f operator'''
        return [self.deltax(clm_co),self.deltay(clm_co),self.deltaz(clm_co)]
    def get_Orientation_Tensor_A2(self,clm_co):
        '''notice should be payed here: the 4Pi normalization factor is different from different normolization method!
        in this package, the 4Pi normolization method is used, however, in mathematica is 2sqrt(Pi) normalization.'''
        degree=self.degree
        c00=clm_co[0,degree]
        c20=clm_co[2,degree]
        c21=clm_co[2,degree+1]
        c22=clm_co[2,degree+2]
        c2minus1=clm_co[2,degree-1]
        c2minus2=clm_co[2,degree-2]
        A11=4*np.pi*(c00/3+c2minus2/np.sqrt(30)-c20/3/np.sqrt(5)+c22/np.sqrt(30))
        A12=4*np.pi*complex(0,1)/np.sqrt(30)*(c2minus2-c22)
        A13=4*np.pi/np.sqrt(30)*(c2minus1-c21)
        A22=4*np.pi*(c00/3-c2minus2/np.sqrt(30)-c20/3/np.sqrt(5)-c22/np.sqrt(30))
        A23=4*np.pi*complex(0,1)/np.sqrt(30)*(c2minus1+c21)
        A33=4*np.pi*(c00/3+2/3/np.sqrt(5)*c20)
        return np.real(np.array([[A11,A12,A13],[A12,A22,A23],[A13,A23,A33]]))
    def _get_Orientation_Tensor_A2(self,clm_co):
        '''notice should be payed here: the 4Pi normalization factor is different from different normolization method!
        in this package, the 4Pi normolization method is used, however, in mathematica is 2sqrt(Pi) normalization.
        
        A11, A12, A13, A22, A23, A33=c . [c00, c20, c21, c22, c2minus1, c2minus2]
        '''
        degree=self.degree
        c00=clm_co[0,degree]
        c20=clm_co[2,degree]
        c21=clm_co[2,degree+1]
        c22=clm_co[2,degree+2]
        c2minus1=clm_co[2,degree-1]
        c2minus2=clm_co[2,degree-2]
        c=np.array(
                [[4.1887902047863909846,-1.8732839282775270382,0,2.2942948838181905767,0,2.2942948838181905767], \
                 [0,0,0,-2.2942948838181905767j,0,2.2942948838181905767j], \
                 [0,0,-2.2942948838181905767,0,2.2942948838181905767,0], \
                 [4.1887902047863909846,-1.8732839282775270382,0,-2.2942948838181905767,0,-2.2942948838181905767], \
                 [0,0,2.2942948838181905767j,0,2.2942948838181905767j,0], \
                 [4.1887902047863909846,3.7465678565550540764,0,0,0,0]])
        A11,A12,A13,A22,A23,A33=np.dot(c,np.array([c00,c20,c21,c22,c2minus1,c2minus2]))
        return np.real(np.array([[A11,A12,A13],[A12,A22,A23],[A13,A23,A33]]))
        
    def get_Orientation_Tensor_A4_fast(self,clm_co):
        '''notice should be payed here: the 4Pi normalization factor is different from different normolization method!
        in this package, the 4Pi normolization method is used, however, in mathematica is 2sqrt(Pi) normalization.'''
        degree=self.degree
        c00=clm_co[0,degree]
        c20=clm_co[2,degree]
        c21=clm_co[2,degree+1]
        c22=clm_co[2,degree+2]
        c2minus1=clm_co[2,degree-1]
        c2minus2=clm_co[2,degree-2]
        c40=clm_co[4,degree]
        c41=clm_co[4,degree+1]
        c42=clm_co[4,degree+2]
        c43=clm_co[4,degree+3]
        c44=clm_co[4,degree+4]
        c4minus1=clm_co[4,degree-1]
        c4minus2=clm_co[4,degree-2]
        c4minus3=clm_co[4,degree-3]
        c4minus4=clm_co[4,degree-4]
        x4=0.2*c00+ 0.156492159287190318130562795086*c2minus2 \
            -0.127775312999987982651952781070*c20+0.156492159287190318130562795086*c22+ \
            0.0398409536444797879989605726564*c4minus4+0.0285714285714285714285714285714*c40+ \
            0.0398409536444797879989605726564*c44-0.0301169300968417079237989861375*c4minus2- \
            0.0301169300968417079237989861375*c42
        y4=0.2*c00-0.1564921592871903181306*c2minus2-0.1277753129999879826520*c20- \
            0.1564921592871903181306*c22+0.03984095364447978799896*c4minus4+0.03011693009684170792380*c4minus2+ \
            0.02857142857142857142857*c40+0.03011693009684170792380*c42+0.03984095364447978799896*c44
        z4=0.2*c00+0.2555506259999759653039*c20+0.07619047619047619047619*c40
        x3y=0.07824607964359515906528j*c2minus2-0.07824607964359515906528j*c22+0.03984095364447978799896j*c4minus4- \
            0.01505846504842085396190j*c4minus2+0.01505846504842085396190j*c42-0.03984095364447978799896j*c44
        x3z=0.07824607964359515906528*c2minus1-0.07824607964359515906528*c21+0.02817180849095055258365*c4minus3- \
            0.03194382824999699566299*c4minus1+0.03194382824999699566299*c41-0.02817180849095055258365*c43
        y3z=0.07824607964359515906528j*c2minus1+0.07824607964359515906528j*c21-0.02817180849095055258365j*c4minus3- \
            0.03194382824999699566299j*c4minus1-0.03194382824999699566299j*c41-0.02817180849095055258365j*c43
        y3x=0.078246079643595159065j*c2minus2-0.078246079643595159065j*c22-0.039840953644479787999j*c4minus4- \
            0.015058465048420853962j*c4minus2+0.015058465048420853962j*c42+0.039840953644479787999j*c44
        z3y=0.078246079643595159065j*c2minus1+0.078246079643595159065j*c21+ \
            0.042591770999995994217j*c4minus1+0.042591770999995994217j*c41
        z3x=0.07824607964359515906528*c2minus1-0.07824607964359515906528*c21+ \
            0.04259177099999599421732*c4minus1-0.04259177099999599421732*c41
        x2y2=0.06666666666666666666667*c00-0.04259177099999599421732*c20-0.03984095364447978799896*c4minus4+ \
            0.009523809523809523809524*c40-0.03984095364447978799896*c44
        x2z2=0.06666666666666666666667*c00+0.02608202654786505302176*c2minus2+0.02129588549999799710866*c20+ \
            0.02608202654786505302176*c22+0.03011693009684170792380*c4minus2-0.03809523809523809523810*c40+ \
            0.03011693009684170792380*c42
        y2z2=0.06666666666666666666667*c00-0.02608202654786505302176*c2minus2+0.02129588549999799710866*c20- \
            0.02608202654786505302176*c22-0.03011693009684170792380*c4minus2-0.03809523809523809523810*c40- \
            0.03011693009684170792380*c42
        x2yz=0.026082026547865053022j*c2minus1+0.026082026547865053022j*c21+0.028171808490950552584j*c4minus3- \
            0.010647942749998998554j*c4minus1-0.010647942749998998554j*c41+0.028171808490950552584j*c43
        y2xz=0.026082026547865053022*c2minus1-0.026082026547865053022*c21-0.028171808490950552584*c4minus3- \
            0.010647942749998998554*c4minus1+0.010647942749998998554*c41+0.028171808490950552584*c43
        z2xy=0.026082026547865053022j*c2minus2-0.026082026547865053022j*c22+ \
            0.030116930096841707924j*c4minus2-0.030116930096841707924j*c42
        '''
        x4=1/5*c00+1/7*np.sqrt(6/5)*c2minus2-2*c20/7/np.sqrt(5)+1/7*np.sqrt(6/5)*c22+c4minus4/3/np.sqrt(70)- \
            1/21*np.sqrt(2/5)*c4minus2+c40/35-c42/21*np.sqrt(2/5)+c44/3/np.sqrt(70)
        y4=c00/5-c2minus2/7*np.sqrt(6/5)-2*c20/7/np.sqrt(5)-c22/7*np.sqrt(6/5)+c4minus4/3/np.sqrt(70)+ \
            c4minus2/21*np.sqrt(2/5)+c40/35+c42/21*np.sqrt(2/5)+c44/3/np.sqrt(70)
        z4=c00/5+4*c20/7/np.sqrt(5)+8/105*c40
        x3y=1j/7*np.sqrt(0.3)*c2minus2-1j/7*np.sqrt(3/10)*c22+1j*c4minus4/3/np.sqrt(70)-1j*c4minus2/21/np.sqrt(10)+ \
            1j*c42/21/np.sqrt(10)-1j*c44/3/np.sqrt(70)
        x3z=c2minus1/7*np.sqrt(3/10)-c21/7*np.sqrt(3/10)+c4minus3/6/np.sqrt(35)-c4minus1/14/np.sqrt(5)+c41/14/np.sqrt(5)- \
            c43/6/np.sqrt(35)
        y3z=1j/7*np.sqrt(3/10)*c2minus1+1j/7*np.sqrt(3/10)*c21-1j*c4minus3/6/np.sqrt(35)-1j*c4minus1/14/np.sqrt(5)- \
            1j*c41/14/np.sqrt(5)-1j*c43/6/np.sqrt(35)
        y3x=c2minus2/7j*np.sqrt(3/10)-1j/7*np.sqrt(3/10)*c22-c4minus4/3j/np.sqrt(70)-c4minus2/21j/np.sqrt(10)+ \
            c42/21j/np.sqrt(10)+c44/3j/np.sqrt(70)
        z3y=c2minus1/7j*np.sqrt(3/10)+c21/7j*np.sqrt(3/10)+2j*c4minus1/21/np.sqrt(5)+2j*c41/21/np.sqrt(5)
        z3x=c2minus1/7*np.sqrt(3/10)-c21/7*np.sqrt(3/10)+2*c4minus1/21/np.sqrt(5)-2*c41/21/np.sqrt(5)
        x2y2=c2minus1/7*np.sqrt(3/10)-c21/7*np.sqrt(3/10)+2*c4minus1/21/np.sqrt(5)-2*c41/21/np.sqrt(5)
        x2z2=c00/15+c2minus2/7/np.sqrt(30)+c20/21/np.sqrt(5)+c22/7/np.sqrt(30)+c4minus2/21*np.sqrt(2/5)- \
            4*c40/105+c42/21*np.sqrt(2/5)
        y2z2=c00/15-c2minus2/7/np.sqrt(30)+c20//21/np.sqrt(5)-c22/7/np.sqrt(30)-c4minus2/21*np.sqrt(2/5) \
            -4*c40/105- c42/21*np.sqrt(2/5)
        x2yz=1j*c2minus1/7/np.sqrt(30)+1j*c21/7/np.sqrt(30)+1j*c4minus3/6/np.sqrt(35)-1j*c4minus1/42/np.sqrt(5)- \
            1j*c41/42/np.sqrt(5)+1j*c43/6/np.sqrt(35)
        y2xz=c2minus1/7/np.sqrt(30)-c21/7/np.sqrt(30)-c4minus3/6/np.sqrt(35)- \
            c4minus1/42/np.sqrt(5)+c41/42/np.sqrt(5)+c43/6/np.sqrt(35)
        z2xy=1j*c2minus2/7/np.sqrt(30)-1j*c22/7/np.sqrt(30)+1j/21*np.sqrt(2/5)*c4minus2-1j/21*np.sqrt(2/5)*c42
        '''
        A4=np.zeros([3,3,3,3])
        A4[0,0,:,:]=np.array([[x4,x3y,x3z],[x3y,x2y2,x2yz],[x3z,x2yz,x2z2]])
        A4[0,1,:,:]=np.array([[x3y,x2y2,x2yz],[x2y2,y3x,y2xz],[x2yz,y2xz,z2xy]])
        A4[0,2,:,:]=np.array([[x3z,x2yz,x2z2],[x2yz,y2xz,z2xy],[x2z2,z2xy,z3x]])
        A4[1,0,:,:]=A4[0,1,:,:]
        A4[1,1,:,:]=np.array([[x2y2,y3x,y2xz],[y3x,y4,y3z],[y2xz,y3z,y2z2]])
        A4[1,2,:,:]=np.array([[x2yz,y2xz,z2xy],[y2xz,y3z,y2z2],[z2xy,y2z2,z3y]])
        A4[2,0,:,:]=A4[0,2,:,:]
        A4[2,1,:,:]=A4[1,2,:,:]
        A4[2,2,:,:]=np.array([[x2z2,z2xy,z3x],[z2xy,y2z2,z3y],[z3x,z3y,z4]])
        # the A4 should be normolized by 4Pi
        return np.real(4*np.pi*A4)
    
    def get_Orientation_Tensor_A4(self,clm_co):
        '''notice should be payed here: the 4Pi normalization factor is different from different normolization method!
        in this package, the 4Pi normolization method is used, however, in mathematica is 2sqrt(Pi) normalization.'''
        degree=self.degree
        x=self.x
        y=self.y
        z=self.z
        x4=x(x(x(x(clm_co))))[0,degree]
        y4=y(y(y(y(clm_co))))[0,degree]
        z4=z(z(z(z(clm_co))))[0,degree]
        
        x3y=y(x(x(x(clm_co))))[0,degree]
        x3z=z(x(x(x(clm_co))))[0,degree]
        y3z=z(y(y(y(clm_co))))[0,degree]
        y3x=x(y(y(y(clm_co))))[0,degree]
        z3y=y(z(z(z(clm_co))))[0,degree]
        z3x=x(z(z(z(clm_co))))[0,degree]
        
        x2y2=y(y(x(x(clm_co))))[0,degree]
        x2z2=z(z(x(x(clm_co))))[0,degree]
        y2z2=z(z(y(y(clm_co))))[0,degree]
        
        x2yz=z(y(x(x(clm_co))))[0,degree]
        y2xz=x(z(y(y(clm_co))))[0,degree]
        z2xy=x(y(z(z(clm_co))))[0,degree]
        
        A4=np.zeros([3,3,3,3])
        A4[0,0,:,:]=np.array([[x4,x3y,x3z],[x3y,x2y2,x2yz],[x3z,x2yz,x2z2]])
        A4[0,1,:,:]=np.array([[x3y,x2y2,x2yz],[x2y2,y3x,y2xz],[x2yz,y2xz,z2xy]])
        A4[0,2,:,:]=np.array([[x3z,x2yz,x2z2],[x2yz,y2xz,z2xy],[x2z2,z2xy,z3x]])
        A4[1,0,:,:]=A4[0,1,:,:]
        A4[1,1,:,:]=np.array([[x2y2,y3x,y2xz],[y3x,y4,y3z],[y2xz,y3z,y2z2]])
        A4[1,2,:,:]=np.array([[x2yz,y2xz,z2xy],[y2xz,y3z,y2z2],[z2xy,y2z2,z3y]])
        A4[2,0,:,:]=A4[0,2,:,:]
        A4[2,1,:,:]=A4[1,2,:,:]
        A4[2,2,:,:]=np.array([[x2z2,z2xy,z3x],[z2xy,y2z2,z3y],[z3x,z3y,z4]])
        # the A4 should be normolized by 4Pi
        return np.real(4*np.pi*A4)

    def to_clm_flatten_co(self,clm_co):
        '''get the flattened coefficients for calculation'''
        return clm_co.flatten()
    def _generate_operator_matrix(self):
        '''
        the basic idea is that:
        exract the linear operators out, and we can get the total operator.
        generate the different operators of matrix type, 
        which act on the the flatten type of spherical harmonics coeeficients.
        the right vector can be generate by:
        clm_flatten_co=np.reshape(clm_co,[-1,])'''
        def _lplus_matrix():
            lplus_PreCo_flat=np.reshape(self.lplus_PreCo,[-1])
            res=sparse.diags(lplus_PreCo_flat[1:],offsets=1)
            return res.tocsr()
        def _lminus_matrix():
            lminus_PreCo_flat=np.reshape(self.lminus_PreCo,[-1])
            res=sparse.diags(lminus_PreCo_flat[:-1],offsets=-1)
            return res.tocsr()
        def _z_matrix():
            zminus_PreCo_flat=np.reshape(self.zminus_PreCo,[-1])
            zminus_res=sparse.diags(zminus_PreCo_flat[2*self.degree+1:],offsets=2*self.degree+1)
            zplus_PreCo_flat=np.reshape(self.zplus_PreCo,[-1])
            zplus_res=sparse.diags(zplus_PreCo_flat[:-2*self.degree-1],offsets=-2*self.degree-1)
            res=zplus_res+zminus_res
            return res.tocsr()
        def _Lz_matrix():
            return sparse.diags(np.reshape(-self.m,[-1])).tocsr()
        def _L2_matrix():
            return sparse.diags(np.reshape(self.L2_PreCo,[-1])).tocsr()
        def _Lx_matrix():
            return 0.5*(self.lminus_matrix+self.lplus_matrix)
        def _Ly_matrix():
            return complex(0,-0.5)*(-self.lminus_matrix+self.lplus_matrix)
        def _x_matrix():
            return complex(0,1)*self.z_matrix.dot(self.Ly_matrix)-complex(0,1)*self.Ly_matrix.dot(self.z_matrix)
        def _y_matrix():
            return complex(0,1)*self.Lx_matrix.dot(self.z_matrix)-complex(0,1)*self.z_matrix.dot(self.Lx_matrix)
        def _deltax_matrix():
            return complex(0,1)*self.z_matrix.dot(self.Ly_matrix)-complex(0,1)*self.y_matrix.dot(self.Lz_matrix)
        def _deltay_matrix():
            return complex(0,1)*self.x_matrix.dot(self.Lz_matrix)-complex(0,1)*self.z_matrix.dot(self.Lx_matrix)
        def _deltaz_matrix():
            return complex(0,1)*self.y_matrix.dot(self.Lx_matrix)-complex(0,1)*self.x_matrix.dot(self.Ly_matrix)
        self.lplus_matrix=_lplus_matrix()
        self.lminus_matrix=_lminus_matrix()
        self.z_matrix=_z_matrix()
        self.Lz_matrix=_Lz_matrix()
        self.L2_matrix=_L2_matrix()
        self.Lx_matrix=_Lx_matrix()
        self.Ly_matrix=_Ly_matrix()
        self.x_matrix=self.revmove_zeros(_x_matrix())
        self.y_matrix=self.revmove_zeros(_y_matrix())
        self.deltax_matrix=self.revmove_zeros(_deltax_matrix())
        self.deltay_matrix=self.revmove_zeros(_deltay_matrix())
        self.deltaz_matrix=self.revmove_zeros(_deltaz_matrix())
        
        def _Identity_matrix():
            Identity=np.zeros_like(self.n)
            Identity[0,self.degree]=1
            return sparse.diags(Identity.flatten()).tocsr()
        self.Identity_matrix=_Identity_matrix()
    def I_uu_dot_matrix_op(self,spase_op_list):
        '''
        input: flattened clm_coefficients
        (I-uu)•f operator
        can be write as:f-u(u•f)
        {fx, fy, fz}-{x,y,z}*(x*fx+y*fy+z*fz)
        =
        {fx-(u•f)*x,
        fy-(u•f)*y,
        fz-(u•f)*z}
        '''
        fx=spase_op_list[0]
        fy=spase_op_list[1]
        fz=spase_op_list[2]
        x=self.x_matrix
        y=self.y_matrix
        z=self.z_matrix
        
        uf=x.dot(fx)+y.dot(fy)+z.dot(fz)
        res=[fx-x.dot(uf), fy-y.dot(uf), fz-z.dot(uf)]
        
        return [self.revmove_zeros(res[0]),self.revmove_zeros(res[1]),self.revmove_zeros(res[2])]
    def revmove_zeros(self,sparse_data):
        '''
        input: sparse matrix
        remove the data less than 1e-15, which is zero actually'''
        real_part=scipy.real(sparse_data)
        imag_part=scipy.imag(sparse_data)
        if len(np.where(np.logical_not(np.abs(real_part.data)<1e-15))[0])!=0:
            real_part.data[np.abs(real_part.data)<1e-15]=0
        if len(np.where(np.logical_not(np.abs(imag_part.data)<1e-15))[0])!=0:
            imag_part.data[np.abs(imag_part.data)<1e-15]=0
        return (real_part+imag_part*1j).tocsr()
    
    def A2_to_clm_co(self,A2):
        '''
        4Pi normolized 
        convert A2 to clm_co
        {A11, A12, A13, A22, A23, A33}=4*np.pi*c.{y00, y20, y21, y22, y2minus1, y2minus2}
        {y00, y20, y21, y22, y2minus1, y2minus2}=inv_c.{A11, A12, A13, A22, A23, A33}/(4*np.pi)
        '''
        inv_c=np.array([[0.079577471545947667884,0,0,0.079577471545947667884,0,0.079577471545947667884], \
                        [-0.088970317927147132310,0,0,-0.088970317927147132310,0,0.17794063585429426462], \
                        [0,0,-0.21793188117470520831,0,-0.21793188117470520831j,0], \
                        [0.10896594058735260415,0.21793188117470520831j,0,-0.10896594058735260415,0,0], \
                        [0,0,0.21793188117470520831,0,-0.21793188117470520831j,0], \
                        [0.10896594058735260415,-0.21793188117470520831j,0,-0.10896594058735260415,0,0]])
        A11=A2[0,0]
        A12=A2[0,1]
        A13=A2[0,2]
        A22=A2[1,1]
        A23=A2[1,2]
        A33=A2[2,2]
        y00,y20,y21,y22,y2minus1,y2minus2=np.dot(inv_c,np.array([A11,A12,A13,A22,A23,A33]))
        clm_co=np.zeros([self.degree+1,2*self.degree+1],np.complex128)
        clm_co[0,self.degree]=y00
        clm_co[2,self.degree]=y20
        clm_co[2,self.degree+1]=y21
        clm_co[2,self.degree+2]=y22
        clm_co[2,self.degree-1]=y2minus1
        clm_co[2,self.degree-2]=y2minus2
        return clm_co