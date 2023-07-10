import numpy as np
import pandas as pd
C=[[-0.1480648093,-0.2106349673,0.4868019601],
    [0.8084618453,0.9092350296,0.5776328438],
    [0.3722003446,-1.2840654776,-2.2462007509],
    [0.7765597096,1.1104441966,0.4605743789],
    [-1.3431772379,0.1260059291,-1.9088154281],
    [-1.7366749542,-2.5375632310,-4.8900459209],
    [0.8895946393,1.9988098293,4.0544348937],
    [1.7367571741,1.4863151577,3.8542602127],
    [-0.0324756095,0.5856304774,1.1817992322],
    [0.6631716575,-0.0756740034,0.9512305286]]
C=pd.DataFrame(C,columns=['m=1','m=2','m=3'])

def a4_ORW3(a2):
    '''ORW3 is an Orthotropic closure approximation method
    A11~A66 the the eigenvalues in the eigenvecter system
    so, the a4 should recovered to the eigenvector system
    '''
    lams,R=np.linalg.eigh(a2)
    lams=np.flip(lams,axis=0)
    R=np.flip(R,axis=1)
    #np.dot(np.dot(R,np.diag(lams)),R.T)
    v1,v2,v3=lams
    iv=[1,v1,v1*v1,v2,v2*v2,v1*v2,v1*v1*v2,v1*v2*v2,v1**3,v2**3]
    A11=np.dot(C.loc[:,'m=1'],iv)
    A22=np.dot(C.loc[:,'m=2'],iv)
    A33=np.dot(C.loc[:,'m=3'],iv)
    #solve the function to get the A44,A55,A66
    #[A11,A22,A33]+dot([[0,1,1],[1,0,1],[1,1,0]],[A44,A55,A66])=[a1,a2,a3]
    coef=np.array([[0,1,1],[1,0,1],[1,1,0]])
    A44,A55,A66=np.dot(np.linalg.inv(coef),(np.array([[v1,v2,v3]])-np.array([[A11,A22,A33]])).T).squeeze()
    #a4=np.zeros((3,3,3,3))
    #a4[0,0,0,0]=A11
    #a4[1,1,1,1]=A22
    #a4[2,2,2,2]=A33
    #a4[2,1,1,2]=a4[1,2,2,1]=a4[2,1,2,1]=a4[1,2,1,2]=a4[2,2,1,1]=a4[1,1,2,2]=A44
    #a4[0,2,0,2]=a4[0,2,2,0]=a4[2,0,0,2]=a4[2,0,2,0]=a4[2,2,0,0]=a4[0,0,2,2]=A55
    #a4[1,0,1,0]=a4[1,0,0,1]=a4[0,1,0,1]=a4[0,1,1,0]=a4[1,1,0,0]=a4[0,0,1,1]=A66
    #a4=np.tensordot(np.dot(R,a4),R.T,axes=(1,0))
    #a4=np.dot(np.dot(R,a4),R.T)
    e1=R[:,0]
    e2=R[:,1]
    e3=R[:,2]
    e11=np.tensordot(e1,e1,axes=0)
    e22=np.tensordot(e2,e2,axes=0)
    e33=np.tensordot(e3,e3,axes=0)
    e1111=np.tensordot(e11,e11,axes=0)
    e2222=np.tensordot(e22,e22,axes=0)
    e3333=np.tensordot(e33,e33,axes=0)
    e1122=np.tensordot(e11,e22,axes=0)
    e2233=np.tensordot(e22,e33,axes=0)
    e1133=np.tensordot(e11,e33,axes=0)
    a4=A11*e1111+A22*e2222+A33*e3333+ \
            A44*(e2233+e2233.transpose([3,2,1,0])+e2233.transpose([0,2,1,3])+e2233.transpose([3,1,2,0])+e2233.transpose([0,3,2,1])+e2233.transpose([3,0,1,2]))+ \
            A55*(e1133+e1133.transpose([3,2,1,0])+e1133.transpose([0,2,1,3])+e1133.transpose([3,1,2,0])+e1133.transpose([0,3,2,1])+e1133.transpose([3,0,1,2]))+ \
            A66*(e1122+e1122.transpose([3,2,1,0])+e1122.transpose([0,2,1,3])+e1122.transpose([3,1,2,0])+e1122.transpose([0,3,2,1])+e1122.transpose([3,0,1,2]))
    return a4
