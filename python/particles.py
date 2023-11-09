# particles.py
#

import numpy as np

from interpolations import rateMan
from interpolations import sampMan
from interpolations import probBlock

import threading
import time

def flip(n):
    if n == 0:
        return 1
    elif n == 1: 
        return 0

# Generates a random 8 character hex string
def hash8():
    return '%08x' % np.random.randint(16**8)

class quark:
    def __init__(self, conf, anti=0):
        self.conf = conf
        self.anti = anti
        self.name = hash8()
        # Initial positions are sampled randomly throughout the box
        self.pos = np.random.uniform(3)*conf['L']
        # Inital momentums are sampled from a Boltzmann distribution
        self.mom = np.random.normal(loc=0, scale=np.sqrt(conf['T']/conf['Mb']),size=3)
        # Space partition
        self.XPart = np.floor(pos*conf['NXPart']/conf['L']).astype(int)
        self.mom4 = np.insert(self.mom,0,np.array((np.dot(self.mom, self.mom)+(self.conf['Mb']**2))),axis=0)
    def Xstep(self):
        self.pos = (self.pos + (self.mom/np.sqrt((self.conf['Mb']**2)+np.dot(self.mom,self.mom)))*self.conf['dt'])%self.conf['L']
    def Pstep(self):
        self.mom = self.mom*(1-0.001) #Drag
        self.mom4 = np.insert(self.mom,0,np.array((np.dot(self.mom, self.mom)+(self.conf['Mb']**2))),axis=0) #Update mom4  
    def exchangeStep(self, partners): # partners = [partner_Xs, partner_Ps]
        partner_Xs = partners[0]
        partner_Ps = partners[1]        
        #Here is where we would iterate over all partners to determine recombination
        
        #For now it will just make a random choice with a small probability !!!!!!!!!
        if np.random.uniform() < 0.00001:
            return np.random.randint(len(partner_Xs))-1
        return None
    def posT(self):
        newXP = np.floor(pos*conf['NXPart']/conf['L']).astype(int) 
        if newXP == self.XPart:
            return False, None, None
        else:
            old = self.XPart
            self.XPart = new
            return True, old, new

 
        
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
            self.mom = np.random.normal(loc=0, scale=np.sqrt(conf['T']/conf['M'+self.state]),size=3)/100
            # Initialize constituent quarks
            self.quarks = [quark(conf),quark(conf,anti=1)]
        else:
            self.quarks = quarks
            self.pos = (quarks[0].pos+quarks[1].pos)/2 #Position set to center of mass
            self.mom = quarks[0].mom+quarks[1].mom #Momentum set to sum of momenta ##########!!!!!!!!!!!!!
    def Xstep(self):
        self.pos = (self.pos + (self.mom/np.sqrt((self.conf['M'+self.state]**2)+np.dot(self.mom,self.mom)))*self.conf['dt'])%self.conf['L']
    def Pstep(self):
        self.mom = self.mom*(1-0.01) #tiny drag
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
    def __init__(self, conf, rates, dists):
        self.conf = conf
        self.cl = stepClock(self.conf['tFn'])
        self.time = 0.0
        self.quarks = [quark(conf,anti=n) for i in range(conf['Nbb']) for n in [0,1]]
        self.bounds = [bound(conf) for i in range(conf['NY'])]
        self.rec = []

        self.rates = rates
        self.dists = dists

        self.XParts = [[[{} for i in range(conf['NXPart'])] for j in range(conf['NXPart'])] for k in range(conf['NXPart'])]
        self.quarkD = {tag:qrk.anti for qrk in self.quarks}
        self.XPresCon = [[[]*self.conf['NXPart']]*self.conf['NXPart']]*self.conf['NXPart']
        
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
    def getNQuarkRels(self, n, anti=0):
        return [[np.linalg.norm(quarks[i].pos-quarks[n].pos),np.linalg.norm(quarks[i].mom-quarks[n].mom)] for i in range(len(quarks)) if quarks[i].anti==anti and i!=n and np.linalg.norm(quarks[i].pos-quarks[n].pos) < conf['XCut'] and np.linalg.norm(quarks[i].mom-quarks[n].mom) < conf['prCut']]
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
    def combineQuarks(self, inds, state, p): # inds -> [ind0,ind1]
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
        for qrk in self.quarks:
            qrk.Pstep()
        for bnd in self.bounds:
            bnd.Pstep()
        
        # Exchange-step

        #Regeneration
        
        #QStates = [[self.get0QuarkX(),self.get0QuarkP()],[self.get1QuarkX(),self.get1QuarkP()]]
        QInds = [self.get0QuarkInd(),self.get1QuarkInd()]
        n = 0
        while n < len(self.quarks): #iterate over all unbound quarks
            options = QInds[flip(self.quarks[n].anti)]

            # RGR channel
            RGRpBs = probBlock([probBlock([probBlock(float(self.rates['RGR'][state](( np.linalg.norm(self.quarks[n].pos-self.quarks[op].pos) ,np.linalg.norm(self.quarks[n].mom-self.quarks[op].mom)))*self.conf['dt']),op) for op in options],state) for state in self.conf['StateList']],'RGR')

            # X channel

            # Y channel


            #Collect probablities of different channels together
            pB = probBlock([RGRpBs],'')
            result = pB(np.random.uniform())

            if result != None:
                choice = result[0]
                if QInds[flip(self.quarks[n].anti)][int(choice)] == n:
                    n+=1
                    continue
                
                #Read channel and sample outgoing momentum

                #RGR channel
                if result[2]=='RGR':

                    qE = ((np.linalg.norm(self.quarks[n].mom-self.quarks[choice].mom)**2)/self.conf['M'+result[1]])+self.conf['E'+result[1]] # Gluon energy is fixed by prel
                    CosTg = (np.random.uniform()*2)-1
                    SinTg = np.sqrt(1-(CosTg**2))
                    Phig = np.random.uniform()*np.pi*2
                    qP = np.array([qE*SinTg*np.cos(Phig), qE*SinTg*np.sin(Phig), qE*CosTg])

                self.combineQuarks([n,QInds[flip(self.quarks[n].anti)][choice]],result[1], -qP)

                #QStates = [[self.get0QuarkX(),self.get0QuarkP()],[self.get1QuarkX(),self.get1QuarkP()]]
                QInds = [self.get0QuarkInd(),self.get1QuarkInd()]
                
                
            else:
                n+=1
        
        # Dissociation

        n = 0
        for bnd in self.bounds: # iterate over all bound states
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
                        qtry = np.random.uniform(self.conf['E'+result[0]],self.conf['E'+result[0]]*self.conf['NPts']) 
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


    def step2(self):
        #electric boogaloo

        self.cl.start()

        for tag in self.quarkD.keys():
            self.quarks[tag].Xstep()
        for bnd in self.bounds:
            bnd.Xstep()
        # P-step
        for tag in self.quarkD.keys():
            self.quarks[tag].Pstep()
        for bnd in self.bounds:
            bnd.Pstep()
        
        # Update space partitions
        self.updateXPart()

        # Exchange-step

        # Regeneration

        # Distribute partition updates over conf['NThreads'] number of threads
        # Generate ijk target list for each thread and start

        threadList = []

        
        for i in range(self.conf['NThreads']):
            thread = threading.Thread(target=self.getRE_worker,)
            threadList.append(thread)
            thread.start()
        
        for thread in threadList:
            thread.join()

        # Get events and manage conflicts

        events = getChkdEvents()




        #Dissociation
        












    # Events come down as a list [str(name1+name2), [channel, state]] (with collisions within the same pair already handled)
    # To check collisions across pairs, this reads in all the pairs to ColCheck as {name1:name2 , ...}  and to see if any are used twice, whenever
    # a new one is added ColCheck.keys() and ColCheck.items() are checked, then confilcts can be handled when one returns false
    def getChkdEvents(self):
        events = {}
        ColCheck = {}
        for ev in self.XPresCon[i][j][k] for i in range(self.conf['NXPart']) for j in range(self.conf['NXPart']) for k in range(self.conf['NXPart']):
            if ev[0][:8] not in ColCheck.keys() and ev[0][8:] not in ColCheck.items():
                ColCheck[ev[0][:8]] = ev[0][8:]
                events[ev[0]] = ev[1] # add to event list
            else:
                continue
                # This is where conflicts will be handled but at first pass they will just be dropped !!!!!!!!!!!!!!!!!!! (This is roughly equivalent to how it worked 
                # before with these just never getting rolled in the first place)
        return events



    def recomEvent(event): # these events come as ['name1name2',[state, channel]]

        if event[1][1] == 'RGA':
            














    def updateXPart(self):
        for tag, qrk in self.quarkD.items():
            XPmove, new, old = qrk.posT()
            if XPmove:
                del self.XParts[old[0]][old[1]][old[2]][tag]
                self.XParts[new[0]][new[1]][new[2]][tag] = qrk.anti

    def getRE_worker(self,targetList):
        for tg in targetList:
            self.getRecomEvents(tg[0],tg[1],tg[2])


    # Return recombination events for for XParts[i][j][k]
    def getRecomEvents(self,i,j,k):
        # Gets all  qrk.anti==0 qrks in partition ijk
        inTags = [tag for tag, ant self.XPart[i][j][k].items() if ant==0)
        #Iterates over each box in neighborhood and collects all antiquark tags
        parTags = [tag for tag, ant in self.XParts[t][u][v].items() for t in (i-1,i,i+1) for u in (j-1,j,j+1) for v in (k-1,k,k+1) if ant==1]
        # For each tag in XPart[i][j][k] get all valid pairs of quark-antiquark momentums and separations
        xpPairs = [{tagIn+tagPar:[self.quarkD[tagIn].mom4, self.quarkD[tagPar].mom4, self.quarkD[tagIn].pos, self.quarkD[tagPar].pos]} for tagIn in inTags for tagPar in parTags]
        # Then we need to boost all of these momentum pairs into p1 + p2

        #  SKIPPED 

        ### No Boost
        pairtags = [pairtag for pairtag, xpP in xpPairs.items()]
        inpVars = np.array([[np.linag.norm(xpP[0][1:]-xpP[1][1:]),np.linag.norm(xpP[2]-xpP[3])] for pairtag, xpP in xpPairs.items()])

        # roll recombination in each channel
        # floor(rateFunc(xr,pr)*dt - R) + 1  with (random number)R in (0,1) <-- this should be 1 for a recombination and 0 otherwise
        # RGA channel
        RGAres = {st:np.floor(self.rates['RGA'][st](inpVars[0,:],inpVars[1,:])*self.conf['dt']-np.random(len(pairtags))).astype(int)+1 for st in self.conf['StateList']}


        res = []
        evTagKey = [[st, ch] for ch in ('RGA',) for st in RateRes.keys()] # This tuple of tags should have the same order as in resLine
        for i in range(len(pairtags)):
            resLine = np.array([RateRes[st][i] for RateRes in (RGAres,) for st in RateRes.keys()])
            if np.all(resLine == 0):
                continue
            else:
                for j in range(len(evTagKey)):
                    if resLine[j] == 1:
                        res.append([pairtags[i],evTagKey[j]]) # Get state and channel for sampled events !!! This just takes the first one to avoid conflicts within the same pair 
                        # But you should be able to order this so you always take the higher rate event (there should be some rate ordering for the same x,p pair)
                        break

                #res.append([pairtags[i],resLine])

        self.XPresCon[i][j][k] = res  #This is a list with elements ['name1name2',[state, channel]]



        







    def getOccupations(self):
        return [len(self.bounds),len(self.quarks)]
    def getOccupationRatio(self):
        return 2*len(self.bounds)/(len(self.quarks)+2*len(self.bounds))
    def getMoms(self):
        return [np.linalg.norm(qrk.mom) for qrk in self.quarks]  
    def getMomDist(self):
        data = [np.norm(qrk.mom) for qrk in self.quarks]
        pBins = np.linspace(0,self.conf['pCut'],40)
        hist, bin_edges = np.histogram(data, bins = pBins)
        return hist
    def recLine(self):
        self.rec.append([self.time, self.getOccupationRatio()])
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
