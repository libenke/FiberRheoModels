#import scipy
#import scipy.io as sio
import numpy as np
import pandas as pd
#import matplotlib.pylab as pylab
import itertools

a=[[0.24940908165786e02,-0.497217790110754e00,0.234146291570999e02],
   [-0.435101153160329e03,0.234980797511405e02,-0.412048043372534E03],
   [0.372389335663877E04,-0.391044251397838E03,0.319553200392089E04],
   [0.703443657916476E04,0.153965820593506E03,0.573259594331015E04],
   [0.823995187366106e06,0.152772950743819e06,-0.485212803064813e05],
   [-0.133931929894245e06,-0.213755248785646e04,-0.605006113515592e05],
   [0.880683515327916e06,-0.400138947092812e04,-0.477173740017567e05],
   [-0.991630690741981e07,-0.185949305922308e07,0.599066486689836e07],
   [-0.159392396237307e05,0.296004865275814e04,-0.110656935176569e05],
   [0.800970026849796e07,0.247717810054366e07,-0.460543580680696e08],
   [-0.237010458689252e07,0.101013983339062e06,0.203042960322874e07],
   [0.379010599355267e08,0.732341494213578e07,-0.556606156734835e08],
   [-0.337010820273821e08,-0.147919027644202e08,0.567424911007837e09],
   [0.322219416256417e05,-0.104092072189767e05,0.128967058686204e05],
   [-0.257258805870567e09,-0.635149929624336e08,-0.152752854956514e10],
   [0.214419090344474e07,-0.247435106210237e06,-0.499321746092534e07],
   [-0.449275591851490e08,-0.902980378929272e07,0.132124828143333e09],
   [-0.213133920223355e08,0.724969796807399e07,-0.162359994620983e10],
   [0.157076702372204e10,0.487093452892595e09,0.792526849882218e10],
   [-0.232153488525298e05,0.138088690964946e05,0.466767581292985e04],
   [-0.395769398304473e10,-0.160162178614234e10,-0.128050778279459e11]]
a=pd.DataFrame(a,columns=[3,4,6])
#pd.set_option('precision',14)

#Tijkl=1/24*(Tijkl+Tjikl+Tijlk+Tjilk+Tjilk+Tklij+Tlkij+Tklji+Tlkji+Tikjl+Tkijl+Tiklj+Tkilj+􏰻Tjlik+Tljik+Tjlki+Tljki+Tiljk+Tlijk+Tilkj+Tlikj+Tjkil+Tkjil+Tjkli+Tkjli)
def SymmetricS(T):
    S=np.zeros_like(T)
    for i in itertools.permutations(range(4),4):
        S=S+T.transpose(i)
    return S/24
def a4_IBOF(a2):
    I=np.eye(3)
    e1,e2,e3=np.linalg.eigvals(a2)
    I1=e1+e2+e3
    I2=e1*e2+e1*e3+e2*e3
    I3=e1*e2*e3
    IA=[1,I2,I2*I2,I3,I3*I3,I2*I3,I2*I2*I3,I2*I3*I3,I2**3,I3**3,I2**3*I3,I2*I2*I3*I3, \
        I2*I3**3,I2**4,I3**4,I2**4*I3,I2**3*I3*I3,I2*I2*I3**3,I2*I3**4,I2**5,I3**5]
    beta3=np.dot(a.loc[:,3],IA)
    beta4=np.dot(a.loc[:,4],IA)
    beta6=np.dot(a.loc[:,6],IA)
    beta1=3/5*(-1/7+0.2*beta3*(1/7+4/7*I2+8/3*I3)-beta4*(1/5-8/15*I2-14/15*I3) \
            -beta6*(1/35-24/105*I3-4/35*I2+16/15*I2*I3+8/35*I2*I2))
    beta2=6/7*(1-1/5*beta3*(1+4*I2)+7/5*beta4*(1/6-I2)- \
            beta6*(-1/5+2/3*I3+4/5*I2-8/5*I2*I2))
    beta5=-4/5*beta3-7/5*beta4-6/5*beta6*(1-4/3*I2)
    #calculate a4
    S1=SymmetricS(np.tensordot(I,I,axes=0))
    S2=SymmetricS(np.tensordot(I,a2,axes=0))
    S3=SymmetricS(np.tensordot(a2,a2,axes=0))
    S4=SymmetricS(np.tensordot(I,a2.dot(a2),axes=0))
    S5=SymmetricS(np.tensordot(a2,a2.dot(a2),axes=0))
    S6=SymmetricS(np.tensordot(a2.dot(a2),a2.dot(a2),axes=0))
    return beta1*S1+beta2*S2+beta3*S3+beta4*S4+beta5*S5+beta6*S6
