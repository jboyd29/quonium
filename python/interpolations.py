# interpolations.py
#

import numpy as np
import csv
from scipy.integrate import quad
from scipy.interpolate import CubicSpline



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
            print(r-runtot)
            if r < runtot:
                return pB(r-runtot)+[self.tag]
        return None


# Overlaps

def OvLp(conf, q , state):
    pr = np.sqrt(M*(q-conf['E'+state]))
    eta = conf['alphaS']*conf['M'+state]/(4*conf['NC']*pr)
    aB = 2/(conf['alphaS']*conf['CF']*conf['M'+state])
    if state == '1S':
        return ( ((2**9)(np.pi**2)(eta)(np.power(aB,7))(np.power(pr,2))(1+np.power(eta,2))np.power(2+eta*aB*pr))  /  ((np.power(1+(aB**2)np.power(pr,2),6))(np.exp(2*np.pi*eta)-1)) ) * np.exp(4*eta*np.arctan(2*aB*pr))

    elif state == '2S':
        return 0



# Dissociation Rates

# Real Gluon Absorption

def RGAint(q, conf, state):  # state = '1S', '2S' ... this is just the integrand
    return np.power(q, 3) * np.sqrt(conf['M'+state]*(q-conf['E'+state])) * (1/(np.exp(q/conf['T'])-1)) * (OvLp(conf, q, state))

def getRGArate(conf, state): # this numerically integrates the RGA integrand from Enl to ['ECut']*Enl and multiplies by the prefactor
    res, error = quad(RGAint, conf['E'+state], conf['E'+state]*50, args=(conf, state))
    return res *((conf['alphaS']*conf['M'+state])/(9*(np.pi**2)))

def getRGAdist(conf, state): # this will evaluate the sampling distribution function in the range Enl - ['ECut']*Enl and returns an interpolation (and adding the point I(q=Enl)=0)
    xVals = np.linspace(conf['E'+state], conf['ECut']*conf['E'+state],conf['NPts'])
    woZero = xVals[2:]
    sampledPts = np.concatenate(([0],RGAint(woZero, conf, state)))
    return CubicSpline(xVals,sampledPts)


# Regeneration Rates




# RateMan
# container that will calculate all the rates and interpolations where necessary to get passed around
class rateMan:
    def __init__(self,conf):
        self.conf = conf
        statelist = self.conf['StateList']   # add others here
        self.rates={}
        
        # Generate Real Gluon Absorption rates
        self.rates['RGA'] = {}
        for state in statelist:
            self.rates['RGA'][state] = getRGArate(self.conf, state) # these are just numbers
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
            self.dists['RGA'][state] = getRGAdist(self.conf, state)
    def __getitem__(self, channel) 
        return self.dists[channel]



