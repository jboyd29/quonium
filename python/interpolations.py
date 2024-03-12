# interpolations.py
#

import numpy as np
import csv
from scipy.integrate import quad
from scipy.integrate import nquad
from scipy.interpolate import CubicSpline
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from scipy.optimize import fmin

from scipy.special import kn




# 1D interpolation
class interpol1D:
    def __init__(self,conf,tag):
        self.conf = conf
        self.tag = tag
        self.interp = self.load()
    def __call__(self,x):
        return self.interp(x)
    def load(self):
        data_list = []
        with open('interpolations/'+self.conf[self.tag], 'r', newline='')as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            header = next(reader)
            for row in reader:
                data_list.append(float(row[0]))
        data_array = np.array(data_list)
        return CubicSpline(np.linspace(float(header[0]),float(header[1]),len(data_array)),data_array)

# probability block gets either (a size or a list of blocks) AND a tag to know who it belonged to, 
#potentially you could store the function objects for the results gere instead 
class probBlock:
    def __init__(self, inp, tag):
        self.content=inp
        self.tag = tag
        if type(inp)==float:
            self.size=inp
        elif type(inp)==list:
            self.size=sum([pB.size for pB in inp])
    def __call__(self, r):
        if type(self.content)==float:
            return [self.tag]
        runtot=0
        for pB in self.content:
            runtot+=pB.size
            #print(r-runtot)
            if r < runtot:
                return pB(r-runtot)+[self.tag]
        return None



# General functions

def qFprRG(conf, prel, st): #gluon momentum as a function of pr in RG process
    if 0 == conf['qRGtype']:
        return (np.power(prel, 2)/conf['Mb']) + conf['E'+st]
    elif 1 == conf['qRGtype']:
        En = np.sqrt(np.power(conf['Mb'],2) + np.power(prel,2))
        return En - (np.power(conf['M'+st],2)/(4*En))

def prFqRG(conf, q, st): #relative momentum as a function of gluon q in RG process
    if 0 == conf['qRGtype']:
        return np.sqrt(conf['Mb']*(q-conf['E'+st]))
    elif 1 == conf['qRGtype']:
        return (1/4)*np.sqrt(np.power(q + np.sqrt(np.power(conf['M'+st],2) + np.power(q,2)),2) - np.power(conf['Mb'],2))









# Overlaps

def OvLp(conf, q , state):
    #pr = np.sqrt(conf['M'+'b']*(q-conf['E'+state]))
    pr = prFqRG(conf, q, state)
    #eta = conf['alphaS']*conf['M'+state]/(4*conf['NC']*pr)
    aB = 2/(conf['alphaS']*conf['CF']*conf['M'+'b'])   #Hardcoded alphaS
    #print('etaM:',np.max(eta))
    if conf['MatrElems'] == 0: # Plane wave matrix elements
        if state == '1S':
            #return ( ((2**9)*(np.pi**2)*(eta)*(np.power(aB,7))*(np.power(pr,2))*(1+np.power(eta,2))*np.power(2+eta*aB*pr,2))  /  ((np.power(1+(aB**2)*np.power(pr,2),6))*(np.exp(2*np.pi*eta)-1)) ) * np.exp(4*eta*np.arctan(aB*pr))
            return ((2**10)*np.pi*np.power(aB,7)*np.power(pr,2)) / np.power(1+(np.power(aB,2)*np.power(pr,2)),6)
        elif state == '2S':
            return 0
    elif conf['MatrElems'] == 1: # Coulomb scattering wave matrix elements
        eta = pr*conf['alphaS']*conf['Mb']/(4*conf['NC'])
        if state == '1S':
            return (((2**9)*(np.pi**2)*eta*np.power(pr,2)*(aB**7)*np.power(2+(eta*pr*aB),2)*(1-np.power(eta,2)))/(np.power(1+(np.power(pr,2)*(aB**2)),6)*(np.exp(2*np.pi*eta)-1)))*np.exp(4*eta*np.arctan(pr*aB))


# Dissociation Rates

# Real Gluon Absorption

def RGAint(q, conf, state):  # state = '1S', '2S' ... this is just the integrand
    return np.power(q, 3) * np.sqrt(conf['M'+state]*(q-conf['E'+state])) * (1/(np.exp(q/conf['T'])-1)) * (OvLp(conf, q, state))

def RGAint2(q, gam, conf, st):
    vc = np.sqrt(1-(1/np.power(gam,2)))
    #pr = np.sqrt(conf['M'+'b']*(q-conf['E'+st]))
    pr = prFqRG(conf, q, st)
    return (2*conf['alphaS']*conf['M'+'b']*conf['T']/(9*(np.pi**2)*vc*np.power(gam,2))) * np.power(q,2)*pr*np.log((1-np.exp(-gam*(1+vc)*q/conf['T']))/(1-np.exp(-gam*(1-vc)*q/conf['T']))) * OvLp(conf, q, st) 

def getRGArate2(conf, st):
    ps = np.linspace(0.001,conf['prCut'],conf['NPts'])  #small offset, gam=0 -> rate->inf
    vs = ps/np.sqrt(np.power(ps,2) + np.power(conf['M'+st],2)) 
    gamms = 1/np.sqrt(1-np.power(vs,2)) 
    Rres = []
    for i in range(conf['NPts']):
        res, error = quad(RGAint2, conf['E'+st], conf['E'+st]*conf['ECut'], args=(gamms[i], conf, st)) 
        Rres.append(res)

    #print('intepPoints \n gamms:',gamms,'\n Rres',Rres)
    interp = interp1d(gamms, np.array(Rres), kind='linear', fill_value='extrapolate') #Interpolation
    return interp

def getRGAratePlot(conf,st,ps):
    vs = ps/np.sqrt(np.power(ps,2) + np.power(conf['M'+st],2))
    gams = 1/np.sqrt(1-np.power(vs,2))
    #print('GAMMS:',gams)
    rateVal = getRGArate2(conf,st)(gams)
    return rateVal



def getRGArate(conf, state): # this numerically integrates the RGA integrand from Enl to ['ECut']*Enl and multiplies by the prefactor
    res, error = quad(RGAint, conf['E'+state], conf['E'+state]*conf['ECut'], args=(conf, state))
    return res *((conf['alphaS']*conf['M'+state]*4)/(9*(np.pi**2)))

def getRGAdist(conf, state): # this will evaluate the sampling distribution function in the range Enl - ['ECut']*Enl and returns an interpolation (and adding the point I(q=Enl)=0) and normal
    xVals = np.linspace(conf['E'+state], conf['ECut']*conf['E'+state],conf['NPts'])
    woZero = xVals[1:]
    sampledPts = np.concatenate(([0],RGAint(woZero, conf, state)))
    if eval(conf['ExportRates']):
        with open('../export/rateRGA_'+state+'.tsv', 'w') as f:                 
            np.savetxt(f, sampledPts, delimiter='\t', fmt='%.8f', newline='\n')
    interp = CubicSpline(xVals,-sampledPts) 
    #normalized this for sampling s.t. max(I) = 1
    Imax = -fmin(interp, np.array(2*conf['E'+state]), full_output=0)
    return CubicSpline(xVals,sampledPts/Imax) ### Check on this !!!

def getRGAdist2(conf, st):
    qs = np.linspace(conf['E'+st]+(0.000001*conf['E'+st]), conf['E'+st]*conf['ECut'], conf['NPts']) # <-get q values + small offset
    ps = np.linspace(0.000001, conf['prCut'], conf['NPts']) # momentum values map to v and gamma
    vs = ps / np.sqrt(np.power(ps,2) + np.power(conf['M'+st],2))
    gs = 1/np.sqrt(1-np.power(vs,2))
    sampPts = []
    for i in range(len(gs)):
        Sgi = RGAint2(qs,gs[i],conf, st)
        sampPts.append(Sgi/np.max(Sgi)) # all are normalized st the max is always 1 for all gamma
    interp = RegularGridInterpolator((gs, qs), np.array(sampPts), method='linear', bounds_error = False)
    return interp



# Regeneration Rates

# Real Gluon Radiation

#def RGRsum(x, pr, conf, state): # x-relative spearation pr-relaltive momentum
    #aB = 2/(conf['alphaS']*conf['CF']*conf['M'+state]) 
    #return (conf['gs']/(conf['NC']**2)) * ( np.exp(-(np.power(x,2))/(2*(aB**2))) / ((2*np.pi*(aB**2))**(3/2)) ) * (8/9)*(conf['alphaS'])*np.power(conf['E'+state]+((np.power(pr,2))/conf['M'+state]),3) * (2+(2/(np.exp((conf["E"+state]+(np.power(pr,2)/conf["M"+state]))/conf["T"])+1))) * OvLp(conf,conf['E'+state]+((np.power(pr,2))/conf['M'+state]), state)

def RGRsum2(x, pr, vcm, conf, state): # <- This one is used in sim
    aB = 2/(conf['alphaS']*conf['CF']*conf['M'+'b'])
    #q = conf['E'+state]+((np.power(pr,2))/conf['M'+'b'])
    q = qFprRG(conf, pr, state) 
    g = 1/np.sqrt(1-np.power(vcm,2))
    return (conf['gs']/(conf['NC']**2)) * ( np.exp(-(np.power(x,2))/(2*(aB**2))) / ((2*np.pi*(aB**2))**(3/2)) ) * (8/9)*(conf['alphaS'])*np.power(q,3) * (2+((conf['T']/(g*vcm*q))*np.log((1-np.exp(-g*(1+vcm)*q/conf['T']))/((1-np.exp(-g*(1-vcm)*q/conf['T'])))))) * OvLp(conf, q, state)

def getRGRrate(conf, state):
    prVals = np.linspace(0,conf['prCut'],conf['NPts'])
    woZero = prVals[1:]
    xVals = np.linspace(0,(1.1)*conf['L']*2*(3**(1/2))/conf['NXPart'],conf['NPts']) ### Added a little extra room (1.1)*
    result = RGRsum(xVals[:,None],woZero[None,:],conf,state)
    sampledPts = np.column_stack((np.linspace(0,0,conf['NPts']),result))
    if eval(conf['ExportRates']):
        with open('../export/rateRGR_'+state+'.tsv', 'w') as f:
            np.savetxt(f, sampledPts, delimiter='\t', fmt='%.8f', newline='\n')
    
    return RegularGridInterpolator((xVals,prVals),sampledPts)

def getRGRratePlot(conf, state):
    prVals = np.linspace(0,conf['prCut'],conf['NPts'])
    rVals = [] 
    for i in range(conf['NPts']):
        res, _ = quad(RGRsum2(), 0, conf['L']/conf['NXPart'], args=(conf, state))
        rVals.append(res)

def rEpint(px, py, pz, M, T):
    return np.exp(-np.sqrt(np.power(M,2) + np.power(px,2) + np.power(py,2) + np.power(pz,2))/T)

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

def calcRelExpec(conf):
    res1, error1 = nquad(rEpint, [[-100,100] for i in range(3)], args=(conf['Mb'],conf['T']))
    Neqb = (6*np.power(conf['L'],3))*(1/np.power(np.pi*2,3))*res1
    res2, error2 = nquad(rEpint, [[-100,100] for i in range(3)], args=(conf['M1S'],conf['T']))
    NeqY = (3*np.power(conf['L'],3))*(1/np.power(np.pi*2,3))*res2
    fug = (-Neqb+np.sqrt(np.power(Neqb,2)-(4*NeqY*(-(conf['Nbb']+conf['NY'])))))/(2*NeqY)
    #print('rNeqb:',Neqb)
    #print('rNeqY:',NeqY)
    return NeqY*np.power(fug,2)/((NeqY*np.power(fug,2))+(Neqb*fug))

def calcRelExpec2(conf):
    Neqb = (6*np.power(conf['L'],3))*(1/(np.power(np.pi,2)*2))*np.power(conf['Mb'],2)*conf['T']*kn(2,conf['Mb']/conf['T'])
    NeqY = (3*np.power(conf['L'],3))*(1/(np.power(np.pi,2)*2))*np.power(conf['M1S'],2)*conf['T']*kn(2,conf['M1S']/conf['T']) ##################!!!!!!!!!!!!!! SIN
    fug = (-Neqb+np.sqrt(np.power(Neqb,2)-(4*NeqY*(-(conf['Nbb']+conf['NY'])))))/(2*NeqY)
    #print('rNeqb:',Neqb)
    #print('rNeqY:',NeqY)
    print('rel-Nhid/Ntot:',NeqY*np.power(fug,2)/((NeqY*np.power(fug,2))+(Neqb*fug)) )
    return NeqY*np.power(fug,2)/((NeqY*np.power(fug,2))+(Neqb*fug))


def pMagDist(p, conf, st): # Momentum distribution
    return np.power(p,2)*np.exp(-np.sqrt(np.power(conf['M'+st],2)+np.power(p,2))/conf['T'])
def getMomDist(conf, st):
    NC, _ = quad(pMagDist, 0, conf['prCut'], args=(conf, st)) #Normalization constant
    pPts = np.linspace(0,conf['prCut'],conf['NPts']) #"X"-points
    CDFpts = []# Y-points
    for i in range(conf['NPts']):
        result, _  = quad(pMagDist, 0, pPts[i], args=(conf, st))
        CDFpts.append(result/NC)
    interp = interp1d(np.array(CDFpts), pPts, kind='linear', fill_value='extrapolate')
    return lambda: interp(np.random.random())

    
def getNMomDistPlot(conf, st):
    NC, _ = quad(pMagDist, 0, conf['prCut'], args=(conf, st)) #Normalization constant
    return pMagDist(np.linspace(0,conf['prCut'],conf['NPts']), conf, st)/NC



def binRGRdat(conf, dat):
    binEdges = np.linspace(0,conf['prCut'],conf['NPts']+1)
    binned = [[] for i in range(conf['NPts'])]
    for itm in dat:
        binned[np.floor(conf['NPts']*itm[0]/conf['prCut']).astype(int)].append(itm[1])
    res = []
    for i in range(len(binned)):
        if binned[i] == []:
            res.append([(binEdges[i]+binEdges[i+1])/2, 0, 0])
        else:
            res.append([(binEdges[i]+binEdges[i+1])/2, np.mean(binned[i]), np.std(binned[i])/np.sqrt(len(binned[i]))])
    return np.array(res)


# RateMan
# container that will calculate all the rates and interpolations where necessary to get passed around
class rateMan:
    def __init__(self,conf):
        self.conf = conf
        statelist = self.conf['StateList']   # add others here
        self.rates={}
        
        #Dissociation

        # Generate Real Gluon Absorption rates
        self.rates['RGA'] = {}
        for state in statelist:
            self.rates['RGA'][state] = getRGArate2(self.conf, state) # these are just numbers
        
        #Regeneration

        #Generate Real Gloun Radiation rates
        #self.rates['RGR'] = {}
        #for state in statelist:
            #self.rates['RGR'][state] = getRGRrate(self.conf, state) # these are 2d interpolating functions that take x and pr as args
            #self.rates['RGR'][state] = 



    def __getitem__(self, channel):
        return self.rates[channel]

# SampMan
# similar container that will contain all the sampling distrbutions
class sampMan:
    def __init__(self, conf):
        self.conf = conf
        statelist = self.conf['StateList']
        self.dists = {}

        # Generate Real Gluon absorption sampling distributions
        self.dists['RGA'] = {}
        for state in statelist:
            self.dists['RGA'][state] = getRGAdist2(self.conf, state)
        
        self.dists['Momentum'] = {}
        self.dists['Momentum']['b'] = getMomDist(self.conf, 'b')
        for st in statelist:
            self.dists['Momentum'][st] = getMomDist(self.conf, st)
    def __getitem__(self, channel): 
        return self.dists[channel]


### Debug functions

def getIM(p4):
    return np.sqrt(np.power(p4[0],2) - (p4[1:] @ p4[1:]))


