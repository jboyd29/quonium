# particles.py
#

import numpy as np
from rates import dissociation

def flip(n):
    if n == 0:
        return 1
    elif n == 1: 
        return 0

class quark:
    def __init__(self, conf, anti=0):
        self.conf = conf
        self.anti = anti
        # Initial positions are sampled randomly throughout the box
        self.pos = np.random.uniform(3)*conf['L']
        # Inital momentums are sampled from a Boltzmann distribution
        self.mom = np.random.normal(loc=0, scale=np.sqrt(conf['T']/conf['bMass']),size=3)
    def Xstep(self):
        self.pos = (self.pos + (self.mom/np.sqrt((self.conf['bMass']**2)+np.dot(self.mom,self.mom)))*self.conf['dt'])%self.conf['L']
    def Pstep(self):
        #Check collision
        if np.random.uniform() < np.sqrt(np.dot(self.mom,self.mom))*self.conf['bsig']/self.conf['bMass']:
            #Collision -> update momentum
            self.mom = self.mom + np.random.normal(loc=0, scale=np.sqrt(self.conf['T']/self.conf['bMass']),size=3)/2
    def exchangeStep(self, partners): # partners = [partner_Xs, partner_Ps]
        partner_Xs = partners[0]
        partner_Ps = partners[1]        
        #Here is where we would iterate over all partners to determine recombination
        
        #For now it will just make a random choice with a small probability !!!!!!!!!
        if np.random.uniform() < 0.001:
            return np.random.randint(len(partner_Xs))-1
        return None
 
        
    #def combine(self):
        #del self
    #def __del__(self):
        #return self

class bound:
    def __init__(self, conf, quarks=None):
        self.conf = conf
        if quarks==None:  # Initial(=None) or recombined
            # Initial positions are sampled randomly throughout the box
            self.pos = np.random.uniform(3)*conf['L']
            # Inital momentums are sampled from a Boltzmann distribution
            self.mom = np.random.normal(loc=0, scale=np.sqrt(conf['T']/conf['YMass']),size=3)
            # Initialize constituent quarks
            self.quarks = [quark(conf),quark(conf,anti=1)]
        else:
            self.quarks = quarks
            self.pos = (quarks[0].pos+quarks[1].pos)/2 #Position set to center of mass
            self.mom = quarks[0].mom+quarks[1].mom #Momentum set to sum of momenta ##########!!!!!!!!!!!!!
    def Xstep(self):
        self.pos = (self.pos + (self.mom/np.sqrt((self.conf['YMass']**2)+np.dot(self.mom,self.mom)))*self.conf['dt'])%self.conf['L']
    def Pstep(self):
        #Check collision
        if np.random.uniform() < np.sqrt(np.dot(self.mom,self.mom))*self.conf['bsig']/self.conf['YMass']:
            #Collision -> update momentum
            self.mom = self.mom + np.random.normal(loc=0, scale=np.sqrt(self.conf['T']/self.conf['bMass']),size=3)/2
    def exchangeStep(self): #CURRENTLY UNUSED
        # Here it would take the dissocation rate but for now ot will be random
        if np.random.uniform() < 0.005:
            return True
        return False
    def dissociate(self): # When this function is called it  returns the constituent quarks as [quark, anti]
        #Set constituent quark positions to the position of siddociated bound
        for qrk in self.quarks:
            qrk.pos = self.pos 
        ##############!!!!!!!!!!!!!!! Here is where you would set the momentum but they are just thermally randomized as of now
        for qrk in self.quarks:
            qrk.mom = np.random.normal(loc=0, scale=np.sqrt(self.conf['T']/self.conf['bMass']),size=3)
        return self.quarks
        
class particleList:
    def __init__(self, conf):
        self.conf = conf
        self.time = 0.0
        self.quarks = [quark(conf,anti=n) for i in range(conf['Nbb']) for n in [0,1]]
        self.bounds = [bound(conf) for i in range(conf['NY'])]
        self.rec = []
        self.rates = dissociation(conf)
        
    def getQuarkX(self):
        return [qrk.pos for qrk in self.quarks]
    def getQuarkP(self):
        return [qrk.mom for qrk in self.quarks]
    def get0QuarkX(self):
        return [qrk.pos for qrk in self.quarks if qrk.anti==0]
    def get1QuarkX(self):
        return [qrk.pos for qrk in self.quarks if qrk.anti==1]
    def get0QuarkP(self):
        return [qrk.mom for qrk in self.quarks if qrk.anti==0]
    def get1QuarkP(self):
        return [qrk.pos for qrk in self.quarks if qrk.anti==1]
    def getBoundX(self):
        return [bnd.pos for bnd in self.bounds]
    def getBoundP(self):
        return [bnd.mom for bnd in self.bounds]
    def get0QuarkInd(self):
        res = []
        for i in range(len(self.quarks)):
            if self.quarks[i].anti==0:
                res.append(i)
        return res
    def get1QuarkInd(self):
        res = []
        for i in range(len(self.quarks)):
            if self.quarks[i].anti==1:
                res.append(i)
        return res
    def dissociateBound(self, ind):
        qrks = self.bounds[ind].dissociate()
        del self.bounds[ind]
        for qrk in qrks:
            self.quarks.append(qrk)
    def combineQuarks(self, inds): # inds -> [ind0,ind1]
        bnd = bound(self.conf, quarks=[self.quarks[inds[0]],self.quarks[inds[1]]])
        for ind in sorted(inds, reverse=True):
            del self.quarks[ind]
        self.bounds.append(bnd)
    def step(self):
        #print(len(self.get0QuarkInd()),len(self.get1QuarkInd()))
        # X-step
        for qrk in self.quarks:
            qrk.Xstep()
        for bnd in self.bounds:
            bnd.Xstep()
        # P-step
        for qrk in self.quarks:
            qrk.Pstep()
        for bnd in self.bounds:
            bnd.Pstep()
        # Exchange-step
        QStates = [[self.get0QuarkX(),self.get0QuarkP()],[self.get1QuarkX(),self.get1QuarkP()]]
        QInds = [self.get0QuarkInd(),self.get1QuarkInd()]
        n = 0
        while n < len(self.quarks):
            choice = self.quarks[n].exchangeStep(QStates[flip(self.quarks[n].anti)])
            if choice != None:
                if QInds[flip(self.quarks[n].anti)][choice] == n:
                    n+=1
                    continue
                QStates = [[self.get0QuarkX(),self.get0QuarkP()],[self.get1QuarkX(),self.get1QuarkP()]]
                QInds = [self.get0QuarkInd(),self.get1QuarkInd()]
                
                self.combineQuarks([n,QInds[flip(self.quarks[n].anti)][choice]])
            else:
                n+=1
        n = 0
        for bnd in self.bounds:
            if self.rates.sampleDiss():
                self.dissociateBound(n)
            else:
                n+=1
        self.time += self.conf['dt']
    def getOccupations(self):
        return [len(self.bounds),len(self.quarks)]
    def getOccupationRatio(self):
        return 2*len(self.bounds)/(len(self.quarks)+2*len(self.bounds))
    def recLine(self):
        self.rec.append([self.time,self.getOccupationRatio()])
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    