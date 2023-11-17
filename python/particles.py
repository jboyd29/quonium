# particles.py
#

import numpy as np

from interpolations import rateMan
from interpolations import sampMan
from interpolations import probBlock

from config import stepClock
from config import colr

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
        self.pos = np.random.rand(3)*conf['L']
        # Inital momentums are sampled from a Boltzmann distribution
        self.mom = np.random.normal(loc=0, scale=np.sqrt(conf['T']/conf['Mb']),size=3)
        # Space partition
        self.XPart = np.floor(self.pos*conf['NXPart']/conf['L']).astype(int)
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
        newXP = np.floor(self.pos*self.conf['NXPart']/self.conf['L']).astype(int) 
        if newXP.all() == self.XPart.all():
            return False, None, None
        else:
            oldXP = self.XPart
            self.XPart = newXP
            return True, oldXP, newXP
    def getPosT(self):
        self.XPart = np.floor(self.pos*self.conf['NXPart']/self.conf['L']).astype(int)
        return self.XPart

 
        
    #def combine(self):
        #del self
    #def __del__(self):
        #return self

class bound:
    def __init__(self, conf, quarks=None, state='1S', p=None):
        self.conf = conf
        self.state = state
        if self.state == None:
            print('No state')
        if quarks==None:  # Initial(=None) or recombined
            # Initial positions are sampled randomly throughout the box
            self.pos = np.random.rand(3)*conf['L']
            # Inital momentums are sampled from a Boltzmann distribution
            self.mom = np.random.normal(loc=0, scale=np.sqrt(conf['T']/conf['M'+self.state]),size=3)/100
            # Initialize constituent quarks
            self.quarks = [quark(conf),quark(conf,anti=1)]
            self.name = self.quarks[0].name + self.quarks[1].name
        else:
            self.quarks = quarks
            self.name = self.quarks[0].name+self.quarks[1].name
            self.pos = (quarks[0].pos+quarks[1].pos)/2 #Position set to center of mass
            self.mom = p #Momentum sent down by event sampler
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

        quarksTemp = [quark(conf,anti=n) for i in range(conf['Nbb']) for n in (0,1)] # This is just for initializing quarks
        boundsTemp = [bound(conf) for i in range(conf['NY'])] # This is just for initializing bounds
        self.quarkCon = {qrk.name:qrk for qrk in quarksTemp} # This contains the actual quark objects {tag:quarkObj}
        self.boundCon = {bnd.name:bnd for bnd in boundsTemp} # This contains the actual bound objects {tag:boundObj}


        #self.XParts = [[[{}]*self.conf['NXPart']]*self.conf['NXPart']]*self.conf['NXPart']
        self.XParts = [[[{} for i in range(conf['NXPart'])] for j in range(conf['NXPart'])] for k in range(conf['NXPart'])] # This is a list with 3 indexes that contains a dictionary 
        # with the tags of all quarks and antiquarks in a dictionary {tag:quark.anti}
        self.initXPart()
        self.XPresCon = [[[[] for i in range(self.conf['NXPart'])] for j in range(self.conf['NXPart'])] for k in range(self.conf['NXPart'])] # These are the containers in which the multithreaded recombination puts results

        
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

                self.combinequarks([n,qinds[flip(self.quarks[n].anti)][choice]],result[1], -qp)

                #qstates = [[self.get0quarkx(),self.get0quarkp()],[self.get1quarkx(),self.get1quarkp()]]
                qinds = [self.get0quarkind(),self.get1quarkind()]
                
                
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
        
        #Start step timer
        self.cl.start()

        for tag in self.quarkCon.keys():
            self.quarkCon[tag].Xstep()
        for tag in self.boundCon.keys():
            self.boundCon[tag].Xstep()
        # P-step
        for tag in self.quarkCon.keys():
            self.quarkCon[tag].Pstep()
        for tag in self.boundCon.keys():
            self.boundCon[tag].Pstep()
        
        # Update space partitions
        self.updateXPart()
        #self.initXPart()

        # Exchange-step

        # Regeneration

        # Distribute partition updates over conf['NThreads'] number of threads
        # Generate ijk target list for each thread and start

        targetList = [[] for i in range(self.conf['NThreads'])]
        for i in range(self.conf['NXPart']):
            for j in range(self.conf['NXPart']):
                for k in range(self.conf['NXPart']):
                    targetList[(i+j+k)%self.conf['NThreads']].append([i,j,k])
        threadList = []
        
        #for i in range(len(targetList)):
        #    print('TL'+str(i)+':',targetList[i])
        #self.XPresCon = [[[[] for i in range(self.conf['NXPart'])] for j in range(self.conf['NXPart'])] for k in range(self.conf['NXPart'])]
        
        for i in range(self.conf['NThreads']):
            thread = threading.Thread(target=self.getRE_worker,args=(targetList, i))
            threadList.append(thread)
            thread.start()
        
        for thread in threadList:
            thread.join()

        # Get events and manage conflicts

        events = self.getChkdREvents()

        for event in events.items():
            self.recomEvent(event)




        #Dissociation
        
        #for tag in self.boundCon.keys():
         #   self.checkDissoc(tag)

        Devents = self.getDissocEvs()
        self.doDissocEvs(Devents)


        #print('Boundtags:',self.boundCon.keys())
        # Increment time
        self.time += self.conf['dt']
        # Stop timer
        self.cl.stop()
        












    # Events come down as a list [str(name1+name2), [channel, state]] (with collisions within the same pair already handled)
    # To check collisions across pairs, this reads in all the pairs to ColCheck as {name1:name2 , ...}  and to see if any are used twice, whenever
    # a new one is added ColCheck.keys() and ColCheck.items() are checked, then confilcts can be handled when one returns false
    def getChkdREvents(self):
        events = {}
        ColCheck = {}
        for i in range(self.conf['NXPart']):
            for j in range(self.conf['NXPart']):
                for k in range(self.conf['NXPart']):
                    for ev in self.XPresCon[i][j][k]: 
                        if ev[0][:8] not in ColCheck.keys() and ev[0][8:] not in ColCheck.values():
                            ColCheck[ev[0][:8]] = ev[0][8:]
                            events[ev[0]] = ev[1] # add to event list
                        else:
                            continue
                            # This is where conflicts will be handled but at first pass they will just be dropped !!!!!!!!!!!!!!!!!!! (This is roughly equivalent to how it worked 
                            # before with these just never getting rolled in the first place)
        return events



    def recomEvent(self, ev): # these events come as ['name1name2',[state, channel]]

        if ev[1][1] == 'RGR':
            # No boost scenario
            qE = ((np.linalg.norm(self.quarkCon[ev[0][:8]].mom-self.quarkCon[ev[0][8:]].mom)**2)/self.conf['M'+ev[1][0]])+self.conf['E'+ev[1][0]] # Gluon energy is fixed by prel

            # No boost scenario
            CosTg = (np.random.uniform()*2)-1
            SinTg = np.sqrt(1-(CosTg**2))
            Phig = np.random.uniform()*np.pi*2
            qP = np.array([qE*SinTg*np.cos(Phig), qE*SinTg*np.sin(Phig), qE*CosTg])

        self.combineTags(ev[0], ev[1][0], -qP)

            

    def combineTags(self, tags, state, qP):
        # quarks need to be removed from XParts
        loc1 = self.quarkCon[tags[:8]].XPart
        del self.XParts[loc1[0]][loc1[1]][loc1[2]][tags[:8]]
        loc2 = self.quarkCon[tags[8:]].XPart
        del self.XParts[loc2[0]][loc2[1]][loc2[2]][tags[8:]]

        # create new bound object and pop quarks out of quarkD, and add bound to bound list
        self.boundCon[tags] = bound(self.conf, quarks=[self.quarkCon.pop(tags[:8]),self.quarkCon.pop(tags[8:])], state=state, p=qP)
    
    def getDissocEvs(self):
        res = []
        for tag in self.boundCon.keys():
            RGAprob = self.rates['RGA'][self.boundCon[tag].state]*self.conf['dt']
            disspB = probBlock([probBlock(RGAprob,'RGA')],tag)
            res.append(disspB(np.random.uniform()))
        return res

    def doDissocEvs(self, evs):
        for result in evs:
            if result == None:
                continue
            else:
                #print('tag:',tag)
                print('D:',result)
                if result[0] == 'RGA':
                    resamp = True
                    st = self.boundCon[result[1]].state
                    while resamp:
                        qtry = np.random.uniform(self.conf['E'+st],self.conf['E'+st]*self.conf['NPts']) 
                        if np.random.uniform() < self.dists['RGA'][st](qtry):
                            resamp = False
                    pmag = (qtry - self.conf['E'+st])/self.conf['M'+st]
                    CosThet = (np.random.uniform()*2)-1
                    SinThet = np.sqrt(1-(CosThet)**2)
                    Phi = np.random.uniform()*2*np.pi
                    
                    qCosThet = (np.random.uniform()*2)-1
                    qSinThet = np.sqrt(1-(CosThet)**2)
                    qPhi = np.random.uniform()*2*np.pi
                    
                    
                    pr = np.array([pmag*SinThet*np.cos(Phi),pmag*SinThet*np.sin(Phi),pmag*CosThet])
                    pq = np.array([qtry*qSinThet*np.cos(qPhi),qtry*qSinThet*np.sin(qPhi),qtry*qCosThet])
                    self.dissociateBoundTag(result[1],pr,pq)
                    continue
                elif result[0] == 'X':
                    #Do x channel
                    continue
                elif result[0] == 'Y':
                    #Do y channel
                    continue


    #This function gets a tag and checks AND does the dissociation
    def checkDissoc(self, tag):
        RGAprob = self.rates['RGA'][self.boundCon[tag].state]*self.conf['dt']

        disspB = probBlock([probBlock(RGAprob,'RGA')],tag)

        result = disspB(np.random.uniform())

        if result == None:
            return
        else:
            #print('tag:',tag)
            print('D:',result)
            if result[0] == 'RGA':
                resamp = True
                st = self.boundCon[tag].state
                while resamp:
                    qtry = np.random.uniform(self.conf['E'+st],self.conf['E'+st]*self.conf['NPts']) 
                    if np.random.uniform() < self.dists['RGA'][st](qtry):
                        resamp = False
                pmag = (qtry - self.conf['E'+st])/self.conf['M'+st]
                CosThet = (np.random.uniform()*2)-1
                SinThet = np.sqrt(1-(CosThet)**2)
                Phi = np.random.uniform()*2*np.pi
                   
                qCosThet = (np.random.uniform()*2)-1
                qSinThet = np.sqrt(1-(CosThet)**2)
                qPhi = np.random.uniform()*2*np.pi


                pr = np.array([pmag*SinThet*np.cos(Phi),pmag*SinThet*np.sin(Phi),pmag*CosThet])
                pq = np.array([qtry*qSinThet*np.cos(qPhi),qtry*qSinThet*np.sin(qPhi),qtry*qCosThet])
                self.dissociateBoundTag(tag,pr,pq)
                return
            elif result[1] == 'X':
                #Do x channel
                return
            elif result[1] == 'Y':
                #Do y channel
                return

    def dissociateBoundTag(self, tag, pr, pq):
        #print('DBG0:',len(self.boundCon.keys()))
        qrks = self.boundCon[tag].dissociate(pr, pq)
        #print('DBG1:',len(self.boundCon.keys()))
        del self.boundCon[tag]
        #print('DBG2:',len(self.boundCon.keys()))
        for qrk in qrks:
            self.quarkCon[qrk.name] = qrk
            loc = qrk.getPosT()
            self.XParts[loc[0]][loc[1]][loc[2]][qrk.name] = qrk.anti





    def updateXPart(self):
        for tag, qrk in self.quarkCon.items():
            XPmove, old, new = qrk.posT()
            if XPmove:
                del self.XParts[old[0]][old[1]][old[2]][tag]
                self.XParts[new[0]][new[1]][new[2]][tag] = qrk.anti
    def initXPart(self):
        self.XParts = [[[{} for i in range(self.conf['NXPart'])] for j in range(self.conf['NXPart'])] for k in range(self.conf['NXPart'])]
        for tag, qrk in self.quarkCon.items():
            XPt = np.floor(qrk.pos*self.conf['NXPart']/self.conf['L']).astype(int)
            self.XParts[XPt[0]][XPt[1]][XPt[2]][tag] = qrk.anti

    def getRE_worker(self,targetList,c):
        for tg in targetList[c]:
            self.getRecomEvents(tg[0],tg[1],tg[2])


    # Return recombination events for for XParts[i][j][k]
    def getRecomEvents(self,i,j,k):
        # Gets all  qrk.anti==0 qrks in partition ijk
        inTags = [tag for tag, ant in self.XParts[i][j][k].items() if ant==0]
        #Iterates over each box in neighborhood and collects all antiquark tags
        parTags = [tag for t in (i-1,i,i+1) for u in (j-1,j,j+1) for v in (k-1,k,k+1) for tag, ant in self.XParts[t % self.conf['NXPart']][u % self.conf['NXPart']][v % self.conf['NXPart']].items() if ant==1]
        # For each tag in XPart[i][j][k] get all valid pairs of quark-antiquark momentums and separations
        xpPairs = {tagIn+tagPar:[self.quarkCon[tagIn].mom4, self.quarkCon[tagPar].mom4, self.quarkCon[tagIn].pos, self.quarkCon[tagPar].pos] for tagIn in inTags for tagPar in parTags}
        # Then we need to boost all of these momentum pairs into p1 + p2

        #  SKIPPED 

        ### No Boost
        pairtags = [pairtag for pairtag, xpP in xpPairs.items()]
        inpVars = np.array([[np.linalg.norm(xpP[0][1:]-xpP[1][1:]),np.linalg.norm(xpP[2]-xpP[3])] for pairtag, xpP in xpPairs.items()])

        if np.size(inpVars) == 0:
            self.XPresCon[i][j][k] = []
            return


        #print('hey',inpVars)
        #print('cat',inpVars[:,0])
        #print('thing',self.rates['RGR']['1S']((inpVars[:,0],inpVars[:,1])))
        #print("RGRres", self.rates['RGR']['1S']((inpVars[:,0],inpVars[:,1]))*self.conf['dt']-np.random.rand(len(pairtags)))


        # roll recombination in each channel
        # floor(rateFunc(xr,pr)*dt - R) + 1  with (random number)R in (0,1) <-- this should be 1 for a recombination and 0 otherwise
        # RGR channel
        RGRres = {st:np.floor(self.rates['RGR'][st]((inpVars[:,0],inpVars[:,1]))*self.conf['dt']-np.random.rand(len(pairtags))).astype(int)+1 for st in self.conf['StateList']}

        RateRes = {'RGR':RGRres}


        res = []
        evTagKey = [[st, ch] for ch in ('RGR',) for st in RateRes[ch].keys()] # This tuple of tags should have the same order as in resLine
        for l in range(len(pairtags)):
            resLine = np.array([RateRes[st][l] for RateRes in (RGRres,) for st in RateRes.keys()])
            if np.all(resLine == 0):
                continue
            else:
                for m in range(len(evTagKey)):
                    if resLine[m] == 1:
                        res.append([pairtags[l],evTagKey[m]]) # Get state and channel for sampled events !!! This just takes the first one to avoid conflicts within the same pair 
                        # But you should be able to order this so you always take the higher rate event (there should be some rate ordering for the same x,p pair)
                        break

                #res.append([pairtags[i],resLine])
        #print('Loc:',(i,j,k))
        self.XPresCon[i][j][k] = res  #This is a list with elements ['name1name2',[state, channel]]   







    def getOccupations(self):
        return [len(self.boundCon.keys()),len(self.quarkCon.keys())]
    def getOccupationRatio(self):
        return 2*len(self.boundCon.keys())/(len(self.quarkCon.keys())+2*len(self.boundCon.keys()))
    def getMoms(self):
        return [np.linalg.norm(qrk.mom) for qrk in self.quarks]  
    def getMomDist(self):
        data = [np.norm(qrk.mom) for qrk in self.quarks]
        pBins = np.linspace(0,self.conf['pCut'],40)
        hist, bin_edges = np.histogram(data, bins = pBins)
        return hist
    def recLine(self):
        self.rec.append([self.time, self.getOccupationRatio()])

    def getNinXParts(self):
        tot = 0 
        for i in range(self.conf['NXPart']):
            for j in range(self.conf['NXPart']):
                for k in range(self.conf['NXPart']):
                    tot += len(self.XParts[i][j][k].keys())
        return tot
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
