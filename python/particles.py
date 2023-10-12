# particles.py
#

import numpy as np

from interpolations import rateMan
from interpolations import sampMan
from interpolations import probBlock

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
        if np.random.uniform() < 0.00001:
            return np.random.randint(len(partner_Xs))-1
        return None
 
        
    #def combine(self):
        #del self
    #def __del__(self):
        #return self

class bound:
    def __init__(self, conf, quarks=None, state='1S'):
        self.conf = conf
        self.state = state
        if self.state == None:
            print('No state')
        if quarks==None:  # Initial(=None) or recombined
            # Initial positions are sampled randomly throughout the box
            self.pos = np.random.uniform(3)*conf['L']
            # Inital momentums are sampled from a Boltzmann distribution
            self.mom = np.random.normal(loc=0, scale=np.sqrt(conf['T']/conf['M'+self.state]),size=3)
            # Initialize constituent quarks
            self.quarks = [quark(conf),quark(conf,anti=1)]
        else:
            self.quarks = quarks
            self.pos = (quarks[0].pos+quarks[1].pos)/2 #Position set to center of mass
            self.mom = quarks[0].mom+quarks[1].mom #Momentum set to sum of momenta ##########!!!!!!!!!!!!!
    def Xstep(self):
        self.pos = (self.pos + (self.mom/np.sqrt((self.conf['M'+self.state]**2)+np.dot(self.mom,self.mom)))*self.conf['dt'])%self.conf['L']
    def Pstep(self):
        #Check collision
        if np.random.uniform() < np.sqrt(np.dot(self.mom,self.mom))*self.conf['bsig']/self.conf['M'+self.state]:
            #Collision -> update momentum
            self.mom = self.mom + np.random.normal(loc=0, scale=np.sqrt(self.conf['T']/self.conf['bMass']),size=3)/2
    def exchangeStep(self): #CURRENTLY UNUSED
        # Here it would take the dissocation rate but for now ot will be random
        if np.random.uniform() < 0.005:
            return True
        return False
    def dissociate(self, prel, q): # When this function is called it  returns the constituent quarks as [quark, anti]
        #Set constituent quark positions to the position of dissociated bound
        for qrk in self.quarks:
            qrk.pos = self.pos 
        ##############!!!!!!!!!!!!!!! Here is where you would set the momentum but they are just thermally randomized as of now
        self.quarks[0].mom = prel + q/2
        self.quarks[1].mom = -prel + q/2
        return self.quarks
        
class particleList:
    def __init__(self, conf):
        self.conf = conf
        self.time = 0.0
        self.quarks = [quark(conf,anti=n) for i in range(conf['Nbb']) for n in [0,1]]
        self.bounds = [bound(conf) for i in range(conf['NY'])]
        self.rec = []

        self.rates = rateMan(conf)
        self.dists = sampMan(conf)
        
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
    def dissociateBound(self, ind, pr, pq):
        qrks = self.bounds[ind].dissociate(pr, pq)
        del self.bounds[ind]
        for qrk in qrks:
            self.quarks.append(qrk)
    def combineQuarks(self, inds, state): # inds -> [ind0,ind1]
        bnd = bound(self.conf, quarks=[self.quarks[inds[0]],self.quarks[inds[1]]], state=state)
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
        #for qrk in self.quarks:
        #    qrk.Pstep()
        #for bnd in self.bounds:
        #    bnd.Pstep()
        # Exchange-step
        QStates = [[self.get0QuarkX(),self.get0QuarkP()],[self.get1QuarkX(),self.get1QuarkP()]]
        QInds = [self.get0QuarkInd(),self.get1QuarkInd()]
        n = 0
        while n < len(self.quarks):
            options = QStates[flip(self.quarks[n].anti)]

            # RGR channel
            RGRpBs = probBlock([probBlock([probBlock(self.rates['RGR'][state]( np.linalg.norm(self.quarks[n].pos-self.quarks[op].pos) ,np.linalg.norm(self.quarks[n].mom-self.quarks[op].mom)),op) for op in options],state) for state in self.conf['StateList']],'RGR')

            # X channel

            # Y channel


            pB = probBlock([RGRpBs])
            result = pB(np.random.uniform())

            choice = result[-1]
            if choice != None:
                if QInds[flip(self.quarks[n].anti)][choice] == n:
                    n+=1
                    continue
                
                self.combineQuarks([n,QInds[flip(self.quarks[n].anti)][choice]])

                QStates = [[self.get0QuarkX(),self.get0QuarkP()],[self.get1QuarkX(),self.get1QuarkP()]]
                QInds = [self.get0QuarkInd(),self.get1QuarkInd()]
                
                
            else:
                n+=1
        n = 0
        for bnd in self.bounds:
            # RGA channel
            RGApBs = probBlock([probBlock(self.rates['RGA'][state]*self.conf['dt'],state) for state in self.conf['StateList']],'RGA')
            
            # X channel

            # Y channel


            channelProbs = [RGApBs]
            

            #Collect all the probabilities together
            pB = probBlock(channelProbs,'')
            result = pB(np.random.uniform())
            if result == None:
                n+=1
                continue
            else:
                # Now read what channel so we can sample appropriately
                if result[1] == 'RGA':
                    resamp = True
                    while resamp:
                        qtry = np.random.uniform(self.conf['E'+result[0]],self.conf['E'+result[0]]*self.conf['NPts']) # !!!!!!
                        if np.random.uniform() < self.dists['RGA'][result[0]](qtry):
                            resamp = False
                    pmag = (qtry - self.conf['E'+result[0]])/self.conf['M'+result[0]]
                    CosThet = (np.random.uniform()*2)-1
                    SinThet = np.sqrt(1-(CosThet)**2)
                    Phi = np.random.uniform()*2*np.pi
                        
                    qCosThet = (np.random.uniform()*2)-1
                    qSinThet = np.sqrt(1-(CosThet)**2)
                    qPhi = np.random.uniform()*2*np.pi


                    pr = np.array([pmag*SinThet*np.cos(Phi),pmag*SinThet*np.sin(Phi),pmag*CosThet])
                    pq = np.array([qtry*qSinThet*np.cos(qPhi),qtry*qSinThet*np.sin(qPhi),qtry*qCosThet])
                    self.dissociateBound(n,pr,pq)


        self.time += self.conf['dt']
    def getOccupations(self):
        return [len(self.bounds),len(self.quarks)]
    def getOccupationRatio(self):
        return 2*len(self.bounds)/(len(self.quarks)+2*len(self.bounds))
    def recLine(self):
        self.rec.append([self.time,self.getOccupationRatio()])
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
