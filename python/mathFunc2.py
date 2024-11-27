import numpy as np
import sys
import os
import time
import random

from scipy.integrate import quad
from scipy.integrate import nquad
from scipy.interpolate import CubicSpline
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import fmin
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import threading

from scipy.special import kv
from scipy.special import kn
from scipy.integrate import cumtrapz
from scipy.ndimage import gaussian_filter

from itertools import chain


### CONFIGURATION SETTER ###

# computes constants and interpolations and adds them to conf

def doConfigCalc(conf):
    # SET CONSTANTS
    conf['aB'] = aBF(conf)
    for st in conf['StateList']:
        conf['E'+st] = np.power(conf['alphaS'],2)*conf['Mb']/2 ######## <<<<<<< CHECK THIS!!!!!!!!
        conf['M'+st] = (2*conf['Mb']-conf['E'+st])
    # Set inverse CDF of fB for sampling momentum
    conf['fBinvCDF_b'] = getfBSampDist(conf, conf['Mb'])
    for st in conf['StateList']:
        conf['fBinvCDF_'+st] = getfBSampDist(conf, conf['M'+st]) 
    # Set channel funciton constants
    for ch in conf['ChannelList']:
        if ch == 'RGA':
            conf[ch+'_C'] = RGAconst(conf)
        elif ch == 'IDQ':
            conf[ch+'_C'] = IDQconst(conf)
        elif ch == 'IDG':
            conf[ch+'_C'] = IDGconst(conf)
        elif ch == 'RGR':
            conf[ch+'_C'] = RGRconst(conf)
        elif ch == 'IRQ':
            conf[ch+'_C'] = IRQconst(conf)
        elif ch == 'IRG':
            conf[ch+'_C'] = IRGconst(conf)
    print('Constants set')
    # Do integrations and interpolations
    for ch in conf['ChannelList']:
        for st in conf['StateList']:
            if ch == 'RGA':
                conf[ch+'_rate'+st] = getRGArate(conf, st) #f(v) final rate
                print(ch+'_rate'+st, 'set')
            elif ch == 'IDQ':
                conf[ch+'_rate'+st] = getIDQrate(conf, st) #f(v) final rate
                print(ch+'_rate'+st, 'set')
            elif ch == 'IDG':
                conf[ch+'_rate'+st] = getIDGrate(conf, st) #f(v) final rate
                print(ch+'_rate'+st, 'set')
            elif ch == 'RGR': #Nothing to do here
                conf[ch+'_rateFv'+st] = NintRecom(conf, RGRvpFC, conf['RGR_C'], st)
                print(ch+'_rateFv'+st, 'set')
            elif ch == 'IRQ':
                conf[ch+'_vInt'+st] = getIRQvInt(conf,st)
                print(ch+'_vInt'+st, 'set')
                conf[ch+'_rateFv'+st] = NintRecom(conf, IRQvpFC, conf['IRQ_C'], st)
                print(ch+'_rateFv'+st, 'set')
            elif ch == 'IRG':
                conf[ch+'_vInt'+st] = getIRGvInt(conf,st)
                print(ch+'_vInt'+st, 'set')
                conf[ch+'_rateFv'+st] = NintRecom(conf, IRGvpFC, conf['IRG_C'], st) 
                print(ch+'_rateFv'+st, 'set')

def getpPts(conf):
    return np.linspace(0.001,conf['prCut']/2,conf['NPts'])
def gettauPts(conf):
    return np.linspace(0, conf['tFn']*conf['dt'], conf['tFn'])
                
def aBF(conf):
    return 2/(conf['alphaS']*conf['CF']*conf['M'+'b'])

# Distributons


# Fermi-Dirac distribution
def nF(En,T):
    return 1/(np.exp(En/T)+1)
def nFn(En,T): # normalized to 1
    return (1/(T*np.log(2)))/(np.exp(En/T)+1)

# Bose-Einstein distribution
def nB(En,T):
    return 1/(np.exp(En/T)-1)

# Boltzmann distribution
def fB(p, M, T):
    return np.power(p,2)*np.exp(-np.sqrt(np.power(M,2)+np.power(p,2))/T)*(1/(np.power(M,2)*T*kv(2,M/T)))

# General functions

def qFpr(conf, prel, st): # gluon q as a function of prel
    return (np.power(prel, 2)/conf['Mb']) + conf['E'+st] 
def prFq(conf, q, st): # prel as a function of gluon q
    return np.sqrt(conf['Mb']*(q-conf['E'+st])) 

def vFg(g): # velocity as a function of gamma
    return np.sqrt(1-(1/np.power(g,2)))
def gFv(v): # gamma as a function of velocity
    return 1/np.sqrt(1-np.power(v,2))

def vFp(p, m): # velocity as a function of momentum
    return p/np.sqrt(np.power(p,2) + np.power(m,2))
def pFv(v, m): # momentum as a function of velocity
    return 0

def vFp4(p4):
    return np.linalg.norm(p4[:,1:]/p4[:,0][:,np.newaxis],axis=1)

def mom324(p3, m):
    return np.insert(p3,0,np.sqrt((p3 @ p3)+np.power(m,2)),axis=0)

def TrgSw(c):
    return np.sqrt(1-np.power(c,2))


# Overlaps
def OvLpPr(conf, pr, st):
    #eta = conf['alphaS']*conf['M'+state]/(4*conf['NC']*pr)
    aB = 2/(conf['alphaS']*conf['CF']*conf['M'+'b'])   #Hardcoded alphaS
    if conf['MatrElems'] == 0: # Plane wave matrix elements
        if st == '1S':
            return ((2**10)*np.pi*np.power(aB,7)*np.power(pr,2)) / np.power(1+(np.power(aB,2)*np.power(pr,2)),6)
        elif st == '2S':
            return 0
    elif conf['MatrElems'] == 1: # Coulomb scattering wave matrix elements
        eta = pr*conf['alphaS']*conf['Mb']/(4*conf['NC'])
        if st == '1S':
            return (((2**9)*(np.pi**2)*eta*np.power(pr,2)*(aB**7)*np.power(2+(eta*pr*aB),2)*(1-np.power(eta,2)))/(np.power(1+(np.power(pr,2)*(aB**2)),6)*(np.exp(2*np.pi*eta)-1)))*np.exp(4*eta*np.arctan(pr*aB))

## MC Numerical Integration for IDQ and IDG 

def MCScInt(vc, conf, intF, st): #Numerical integrate inelastic channel for given vc
    NMC = 1000000
    np.random.seed(hash(str((np.sqrt(2)*threading.get_ident())+vc+time.time()))%65532)
    
    pD = [0,10]
    cD = [-1,1]
    phiD = [0,2*np.pi]
    tsL = pD[1]-pD[0]-conf['E'+st]
    p1s = np.random.uniform(0,1,NMC)
    p2s = (1-np.sqrt(np.random.uniform(0,1,NMC)))
    p1s = (p1s*(p2s-1)+1)*tsL + conf['E'+st]
    p2s = p2s * tsL
    c1s = np.random.uniform(cD[0],cD[1],NMC)
    c2s = np.random.uniform(cD[0],cD[1],NMC)
    phi2s = np.random.uniform(phiD[0],phiD[1],NMC)    
    smp = intF(conf,vc,p1s,c1s,p2s,c2s,phi2s,st)
    vol = (1/2)*(pD[1]-pD[0]-conf['E'+st])*(pD[1]-pD[0]-conf['E'+st])*(cD[1]-cD[0])*(cD[1]-cD[0])*(phiD[1]-phiD[0])
    return np.mean(smp)*vol

def MCintFv(conf, intFunc, st):
    pPts = getpPts(conf)
    vPts = vFp(pPts, conf['M'+st])
    lmbd_arg = partial(MCScInt, intF=intFunc, st=st, conf=conf)
    with ProcessPoolExecutor() as executor:
        result = list(executor.map(lmbd_arg,vPts))    
    return np.array([vPts,result])
def MCintFvRecom(conf, intFunc, st):
    pPts = getpPts(conf)
    vPts = vFp(pPts, conf['Mb'])
    lmbd_arg = partial(MCScInt, intF=intFunc, st=st, conf=conf)
    with ProcessPoolExecutor() as executor:
        result = list(executor.map(lmbd_arg,vPts))    
    return np.array([vPts,result])
### DISSOCIATION FUNCTIONS ###

### RGA ###

def RGAconst(conf):
    return (2*conf['alphaS']*conf['Mb']*conf['T'])/(9*np.power(np.pi,2))

def RGAIntg(q, gam, conf, st):
    vc = vFg(gam)
    #pr = np.sqrt(conf['M'+'b']*(q-conf['E'+st]))
    pr = prFq(conf, q, st)
    return (1/(vc*np.power(gam,2)))*np.power(q,2)*pr*np.log((1-np.exp(-gam*(1+vc)*q/conf['T']))/(1-np.exp(-gam*(1-vc)*q/conf['T']))) * OvLpPr(conf, pr, st)

def getRGArate(conf, st):
    ps = getpPts(conf)
    vs = vFp(ps,conf['M'+st])
    gs = gFv(vs)
    Rres = []
    for i in range(conf['NPts']):
        res, error = quad(RGAIntg, conf['E'+st], conf['E'+st]*conf['ECut'], args=(gs[i], conf, st)) 
        Rres.append(conf['RGA_C']*res)
    interp = interp1d(vs, np.array(Rres), kind='linear', fill_value='extrapolate') #Interpolation
    return interp
    

### IDQ ###

def IDQconst(conf):
    return (2*np.power(conf['alphaS'],2)*conf['Mb'])/(9*np.power(np.pi,4))
def IDQdist(conf,p1,c1,p2,c2,vc):
    return p1*p2*nF(p1*(1+(vc*c1))/np.sqrt(1-np.power(vc,2)),conf['T'])*(1-nF(p2*(1+(vc*c2))/np.sqrt(1-np.power(vc,2)),conf['T'])) 
def IDQangle(p1,c1,p2,c2,phi2):
    return p1*p2*((1+(TrgSw(c1)*TrgSw(c2)*np.cos(phi2))+(c1*c2)))/(np.power(p1,2)+np.power(p2,2)-(2*p1*p2*((TrgSw(c1)*TrgSw(c2)*np.cos(phi2))+(c1*c2))))
def IDQint(conf,vc,p1,c1,p2,c2,phi2,st):
    return IDQdist(conf,p1,c1,p2,c2,vc)*IDQangle(p1,c1,p2,c2,phi2)*OvLpPr(conf,np.sqrt(conf["Mb"]*(p1-conf["E"+st]-p2)),st)*np.sqrt(conf["Mb"]*(p1-conf["E"+st]-p2)) 
def getIDQrate(conf, st):
    MCres = MCintFv(conf, IDQint, st)
    g = gFv(MCres[0])
    interp = interp1d(MCres[0], conf['IDQ_C']*gaussian_filter(MCres[1],2)/g, kind='linear', fill_value='extrapolate') #Interpolation
    return interp
    


### IDG ###

def IDGconst(conf):
    return (np.power(conf['alphaS'],2)*conf['Mb'])/(12*np.power(np.pi,4))
def IDGdist(conf,q1,c1,q2,c2,vc):
    return q1*q2*nB(q1*(1+(vc*c1))/np.sqrt(1-np.power(vc,2)),conf['T'])*(1-nB(q2*(1+(vc*c2))/np.sqrt(1-np.power(vc,2)),conf['T'])) 
def IDGangle(q1,c1,q2,c2,phi2):
    return (np.power(q1,2)+np.power(q2,2))*((1+(TrgSw(c1)*TrgSw(c2)*np.cos(phi2))+(c1*c2)))/(np.power(q1,2)+np.power(q2,2)-(2*q1*q2*((TrgSw(c1)*TrgSw(c2)*np.cos(phi2))+(c1*c2))))
def IDGint(conf,vc,q1,c1,q2,c2,phi2,st):
    return IDGdist(conf,q1,c1,q2,c2,vc)*IDGangle(q1,c1,q2,c2,phi2)*OvLpPr(conf,np.sqrt(conf["Mb"]*(q1-conf["E"+st]-q2)),st)*np.sqrt(conf["Mb"]*(q1-conf["E"+st]-q2))   
def getIDGrate(conf, st):
    MCres = MCintFv(conf, IDGint, st)
    g = gFv(MCres[0])
    interp = interp1d(MCres[0], conf['IDG_C']*gaussian_filter(MCres[1],2)/g, kind='linear', fill_value='extrapolate') #Interpolation
    return interp  
    
### REGENERATION FUNCTIONS ###

def gaussX(conf, x):
    aB = conf['aB']
    return np.exp(-np.power(x,2)/(2*np.power(aB,2)))/np.power(2*np.pi*np.power(aB,2),3/2)

### RGR ###

def RGRvpF(vc, pr, conf, st):
    aB = conf['aB']
    #q = conf['E'+state]+((np.power(pr,2))/conf['M'+'b'])
    q = qFpr(conf, pr, st) 
    g = gFv(vc)
    return np.power(q,3)*(2+((conf['T']/(g*vc*q))*np.log((1-np.exp(-g*(1+vc)*q/conf['T']))/((1-np.exp(-g*(1-vc)*q/conf['T'])))))) * OvLpPr(conf, pr, st)    
def RGRvpFC(vp, conf, st):
    return RGRvpF(vp[0],vp[1],conf,st)
def RGRconst(conf):
    return (conf['gs']/(np.power(conf['NC'],2)))*(8/9)*(conf['alphaS'])
       
def RGRsumC(x, vc, pr, conf, st):
    return gaussX(conf,x)*RGRvpF(vc,pr,conf,st)*conf['RGR_C']


### IRQ ###

def IRQconst(conf):
    return (conf['gs']/(np.power(conf['NC'],2)))*((8*np.power(conf['alphaS'],2))/(9*np.power(np.pi,2)))
def IRQdist(conf,p1,c1,p2,c2,vc):
    g = gFv(vc)
    return p1*p2*nF(p1*(1+(vc*c1))*g,conf['T'])*(1-nF(p2*(1+(vc*c2))*g,conf['T']))
def IRQint(conf,vc,p1,c1,p2,c2,phi2,st):
    return IRQdist(conf,p1,c1,p2,c2,vc)*IDQangle(p1,c1,p2,c2,phi2)#*OvLpPr(conf,np.sqrt(conf["Mb"]*(p1+conf["E"+st]-p2)),st)#*np.sqrt(conf["Mb"]*(p1-conf["E"+st]-p2)) 
def getIRQvInt(conf, st):
    MCres = MCintFvRecom(conf, IRQint, st) #independent of state
    interp = interp1d(MCres[0], gaussian_filter(MCres[1],2), kind='linear', fill_value='extrapolate') #Interpolation 
    return interp
def IRQvpF(vc, pr, conf, st):
    #aB = conf['aB']
    return conf['IRQ_vInt'+st](vc)*OvLpPr(conf,pr,st)#*np.sqrt(pr) 
def IRQvpFC(vp,conf, st):
    return IRQvpF(vp[0],vp[1],conf,st)

def IRQsumC(x, vc, pr, conf, st):
    return gaussX(conf,x)*IRQvpF(vc,pr,conf,st)*conf['IRQ_C']


### IRG

def IRGconst(conf):
    return (conf['gs']/(np.power(conf['NC'],2)))*(np.power(conf['alphaS'],2)/(3*np.power(np.pi,2)))
def IRGdist(conf,q1,c1,q2,c2,vc):
    g = gFv(vc)
    return q1*q2*nB(q1*(1+(vc*c1))*g,conf['T'])*(1-nB(q2*(1+(vc*c2))*g,conf['T']))
def IRGint(conf,vc,q1,c1,q2,c2,phi2,st):
    return IRGdist(conf,q1,c1,q2,c2,vc)*IDGangle(q1,c1,q2,c2,phi2)#*OvLpPr(conf,np.sqrt(conf["Mb"]*(q1-conf["E"+st]-q2)),st) 
def getIRGvInt(conf, st):
    MCres = MCintFvRecom(conf, IRGint, st) #independent of state
    interp = interp1d(MCres[0], gaussian_filter(MCres[1],2), kind='linear', fill_value='extrapolate') #Interpolation 
    return interp
def IRGvpF(vc, pr, conf, st):
    aB = conf['aB']
    return conf['IRG_vInt'+st](vc)*OvLpPr(conf,pr,st) 
def IRGvpFC(vp,conf, st):
    return IRGvpF(vp[0],vp[1],conf,st)

def IRGsumC(x, vc, pr, conf, st):
    return gaussX(conf,x)*IRGvpF(vc,pr,conf,st)*conf['IRG_C']

######## OTHER #########

def sampleSphere(n):
    phi = np.random.uniform(0, 2 * np.pi, n)  # Azimuthal angle
    cos_theta = np.random.uniform(-1, 1, n)   # Cosine of polar angle
    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    points = np.vstack((x, y, z)).T
    return points
    #return np.random.normal(size=(N, 3)); points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
def samplefB(N, conf):
    x = np.linspace(0.001,conf['prCut']/2,1000)
    cdf_values = cumtrapz(fB(x,conf['Mb'],conf['T']), x, initial=0)
    invCDF = interp1d(cdf_values, x, kind='linear', bounds_error=False, fill_value=(0, 1))
    return invCDF(np.random.rand(N))   

def BoostL(v): # v is a vector of velocity vectors vi 
    vx, vy, vz = v.T # get the x y and z components of each vi
    vM = np.linalg.norm(v, axis=1) # get vector of magnitudes of each velocity
    g = 1/np.sqrt(1 - np.power(vM,2)) # gamma
    vvT = np.einsum('ij,ik->ijk',v,v) # batched outer product of v and velocity
    res = np.zeros((len(v),4,4)) # init containers
    res[:, 0, 0] = g # 00 elem set to gamma
    res[:, 0, 1:] = -g[:,np.newaxis]*v # 0j set to -g*v
    res[:, 1:, 0] = -g[:,np.newaxis]*v # i0 to -g*v 
    res[:, 1:, 1:] = np.tile(np.eye(3), (len(v), 1, 1)) + (g[:,np.newaxis,np.newaxis]-np.ones((len(v),3,3)))*vvT/np.power(vM,2)[:,np.newaxis,np.newaxis] # i-123 j-123 set to 1 - (g-1)vvT/vM
    return res
def Vcell(pcm4):
    return pcm4[:,1:]/pcm4[:,0,np.newaxis]
def pDist(L, x1, x2):
    delta = np.abs(x1 - x2)
    periodic_delta = np.minimum(delta, L - delta)
    return np.linalg.norm(periodic_delta, axis=1)

def sampPrelFp1(p1M, conf):
    sN = 10000
    v1 = vFp(p1M, conf['Mb'])
    vi = v1*sampleSphere(sN)
    g = gFv(v1)
    p1 = conf['Mb']*g*np.hstack([np.linspace(1,1,vi.shape[0]).reshape(-1, 1), vi])
    p23 = sampleSphere(sN)*samplefB(sN, conf)[:, np.newaxis]
    p2 = np.hstack([np.sqrt(np.power(p23[:,0],2)+np.power(p23[:,1],2)+np.power(p23[:,2],2)+np.power(conf['Mb'],2)).reshape(-1, 1), p23])
    vcm = (p1[:,1:]+p2[:,1:])/((p1[:,0]+p2[:,0])[:, np.newaxis])
    bL = BoostL(vcm)
    bp1, bp2 = np.einsum('ijk,ik->ij',bL,p1), np.einsum('ijk,ik->ij',bL,p2)
    cmBpr = np.linalg.norm((1/2)*(bp1 - bp2)[:,1:],axis=1)
    vcmM = np.linalg.norm(vcm,axis=1)
    return [vcmM, cmBpr]
    
def NintRecom(conf, vpFunc, Rconst, st):
    pPts = getpPts(conf)
    vPts = vFp(pPts, conf['Mb'])
    res = (1/np.power(conf['L'],3))*Rconst*np.array([np.mean(vpFunc(sampPrelFp1(pi, conf),conf,st)) for pi in pPts])
    interp = interp1d(vPts, res, kind='linear', fill_value='extrapolate') #Interpolation
    return interp

### Expected equilibrium values

def cEpint(px, py, pz, M, T):
    return np.exp(-(M + ((np.power(px,2)+ np.power(py,2) + np.power(pz,2))/(2*M)))/T)

def calcClassExpec(conf):
    res1, error1 = nquad(cEpint, [[-100,100] for i in range(3)], args=(conf['Mb'],conf['T']))
    Neqb = (6*np.power(conf['L'],3))*(1/np.power(np.pi*2,3))*res1
    res2, error2 = nquad(cEpint, [[-100,100] for i in range(3)], args=(conf['M1S'],conf['T']))
    NeqY = (3*np.power(conf['L'],3))*(1/np.power(np.pi*2,3))*res2
    fug = (-Neqb+np.sqrt(np.power(Neqb,2)-(4*NeqY*(-(conf['Nbb']+conf['NY'])))))/(2*NeqY)
    #print('cNeqb:',Neqb)
    #print('cNeqY:',NeqY)
    print('nonrel-Nhid/Ntot:',NeqY*np.power(fug,2)/((NeqY*np.power(fug,2))+(Neqb*fug)) )
    return NeqY*np.power(fug,2)/((NeqY*np.power(fug,2))+(Neqb*fug))

def calcRelExpec2(conf):
    Neqb = (6*np.power(conf['L'],3))*(1/(np.power(np.pi,2)*2))*np.power(conf['Mb'],2)*conf['T']*kn(2,conf['Mb']/conf['T'])
    NeqY = (3*np.power(conf['L'],3))*(1/(np.power(np.pi,2)*2))*np.power(conf['M1S'],2)*conf['T']*kn(2,conf['M1S']/conf['T']) ##################!!!!!!!!!!!!!! SIN
    fug = (-Neqb+np.sqrt(np.power(Neqb,2)-(4*NeqY*(-(conf['Nbb']+conf['NY'])))))/(2*NeqY)
    #print('rNeqb:',Neqb)
    #print('rNeqY:',NeqY)
    print('rel-Nhid/Ntot:',NeqY*np.power(fug,2)/((NeqY*np.power(fug,2))+(Neqb*fug)) )
    return NeqY*np.power(fug,2)/((NeqY*np.power(fug,2))+(Neqb*fug))

def getfBSampDist(conf, M):
    x = np.linspace(0.001,conf['prCut']/2,1000)
    cdf_values = cumtrapz(fB(x,conf['Mb'],conf['T']), x, initial=0)
    invCDF = interp1d(cdf_values, x, kind='linear', bounds_error=False, fill_value=(0, 1))
    return invCDF

def chooseStateInit(conf):
    return random.choice(conf['StateList'])

def getIM(p4):
    return np.sqrt(np.power(p4[0],2) - (p4[1:] @ p4[1:]))
def flattenL(Lst):
    return list(chain(*Lst)) 
