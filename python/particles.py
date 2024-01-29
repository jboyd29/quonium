# particles.py
#

import numpy as np

from interpolations import rateMan
from interpolations import sampMan
from interpolations import probBlock

from interpolations import RGRsum2

from config import stepClock
from config import colr

from hydro import hydroBac

import threading
import time

import multiprocessing


def flip(n):
    if n == 0:
        return 1
    elif n == 1: 
        return 0

# Generates a random 8 character hex string
def hash8():
    return '%08x' % np.random.randint(16**8)

class quark:
    def __init__(self, conf, anti=0, initPmag=1):
        self.conf = conf
        self.anti = anti
        self.name = hash8()
        # Initial positions are sampled randomly throughout the box
        self.pos = np.random.rand(3)*conf['L']
        # Inital momentums are sampled from a Boltzmann distribution
        #self.mom = np.random.normal(loc=0, scale=np.sqrt(conf['T']/conf['Mb']),size=3)
        # Initial momentums are sampled from a Boltzmann distributions
        if 0 == conf['pSampleType']: #Thermal sampling
            momD = np.random.randn(3)
            self.mom = initPmag*momD/np.linalg.norm(momD)
        elif 1 == conf['pSampleType']: #Uniform sampling from pi=0 -> pi=conf["UniPMax"]
            self.mom = np.random.random(3)*conf['UniPMax']
        elif 2 == conf['pSampleType']:
            self.mom = np.array([0,0,conf['UniPMax']])
        # Space partition
        self.XPart = np.floor(self.pos*conf['NXPart']/conf['L']).astype(int)
        self.mom4 = np.insert(self.mom,0,np.array(np.sqrt(np.dot(self.mom, self.mom)+(self.conf['Mb']**2))),axis=0)
    def Xstep(self):
        self.pos = (self.pos + (self.mom/np.sqrt((self.conf['Mb']**2)+np.dot(self.mom,self.mom)))*self.conf['dt'])%self.conf['L']
    def Pstep(self):
        #self.mom = self.mom*(1-0.001) #Drag
        self.mom4 = np.insert(self.mom,0,np.array(np.sqrt(np.dot(self.mom, self.mom)+(self.conf['Mb']**2))),axis=0) #Update mom4  
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
    def __init__(self, conf, quarks=None, state='1S', p=None, initPmag=1):
        self.conf = conf
        self.state = state
        if self.state == None:
            print('No state')
        if quarks==None:  # Initial(=None) or recombined
            # Initial positions are sampled randomly throughout the box
            self.pos = np.random.rand(3)*conf['L']
            # Inital momentums are sampled from a Boltzmann distribution
            if 0 == conf['pSampleType']: #Thermal sampling
                momD = np.random.randn(3)
                self.mom = initPmag*momD/np.linalg.norm(momD)
            elif 1 == conf['pSampleType']: #Uniform sampling from pi=0 -> pi=conf["UniPMax"]
                self.mom = np.random.random(3)*conf['UniPMax']
            elif 2 == conf['pSampleType']:
                self.mom = np.array([0,0,conf['UniPMax']])
            self.mom4 = np.insert(self.mom,0,np.array(np.sqrt(np.dot(self.mom, self.mom)+(self.conf['M'+self.state]**2))),axis=0)
            # Initialize constituent quarks
            self.quarks = [quark(conf),quark(conf,anti=1)]
            self.name = self.quarks[0].name + self.quarks[1].name
        else:
            self.quarks = quarks
            self.name = self.quarks[0].name+self.quarks[1].name
            self.pos = (quarks[0].pos+quarks[1].pos)/2 #Position set to center of mass
            self.mom = p #Momentum sent down by event sampler
            self.mom4 = np.insert(self.mom,0,np.array(np.sqrt(np.dot(self.mom, self.mom)+(self.conf['M'+self.state]**2))),axis=0) 
    def Xstep(self):
        self.pos = (self.pos + (self.mom/np.sqrt((self.conf['M'+self.state]**2)+np.dot(self.mom,self.mom)))*self.conf['dt'])%self.conf['L']
    def Pstep(self):
        #self.mom = self.mom*(1-0.001) #tiny drag
        self.mom4 = np.insert(self.mom,0,np.array(np.sqrt(np.dot(self.mom, self.mom)+(self.conf['M'+self.state]**2))),axis=0)
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

        self.rates = rates
        self.dists = dists
        self.HB = hydroBac(conf)

        #self.quarks = [quark(conf,anti=n,initPmag=dists['Momentum']['b']()) for i in range(conf['Nbb']) for n in [0,1]]
        #self.bounds = [bound(conf, initPmag=dists['Momentum']['1S']()) for i in range(conf['NY'])]
        self.rec = []


        self.recDump = {} #Container for various debug outputs
        self.recDump['RGArateSampDist'] = []

        
        quarksTemp = [quark(conf, anti=n, initPmag=dists['Momentum']['b']()) for i in range(conf['Nbb']) for n in (0,1)] # This is just for initializing quarks
        boundsTemp = [bound(conf, initPmag=dists['Momentum']['1S']()) for i in range(conf['NY'])] # This is just for initializing bounds
        self.quarkCon = {qrk.name:qrk for qrk in quarksTemp} # This contains the actual quark objects {tag:quarkObj}
        self.boundCon = {bnd.name:bnd for bnd in boundsTemp} # This contains the actual bound objects {tag:boundObj}

        #self.XParts = [[[{}]*self.conf['NXPart']]*self.conf['NXPart']]*self.conf['NXPart']
        self.XParts = [[[{} for i in range(conf['NXPart'])] for j in range(conf['NXPart'])] for k in range(conf['NXPart'])] # This is a list with 3 indexes that contains a dictionary 
        # with the tags of all quarks and antiquarks in a dictionary {tag:quark.anti}
        self.initXPart()
        self.XPresCon = [[[[] for i in range(self.conf['NXPart'])] for j in range(self.conf['NXPart'])] for k in range(self.conf['NXPart'])] # These are the containers in which the multithreaded recombination puts results

        
        # Some initial data collection
        self.recDump['RGRrateP1S'] = self.measureRGRrateGam('1S')


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

#26 to #7 - twistedrider Reply+4(4.5 hours)
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
        if 1 == self.conf['doRecom']: # Switch for turning off for recombination
            # Distribute partition updates over conf['NThreads'] number of threads
            # Generate ijk target list for each thread and start

            targetList = [[] for i in range(self.conf['NThreads'])]
            for i in range(self.conf['NXPart']):
                for j in range(self.conf['NXPart']):
                    for k in range(self.conf['NXPart']):
                        targetList[(i+(j*self.conf['NXPart'])+(k*(self.conf['NXPart']**2)))%self.conf['NThreads']].append([i,j,k])
            
            eventsRaw = []
            result_queue = multiprocessing.Queue()
            processList = []
            for i in range(self.conf['NThreads']):
                process = multiprocessing.Process(target=regenUpdate_worker, args=(result_queue, self, targetList[i], np.random.randint(2**32), i))
                processList.append(process)
                process.start()

            while any(process.is_alive() for process in processList) or not result_queue.empty():
                #if len(processList)!=0:
                    #print('remaining:',processList)
                while not result_queue.empty():
                    eventsRaw.append(result_queue.get())
            for process in processList:
                process.join()

            # Get events and manage conflicts
            self.events = self.checkREvents(eventsRaw)
            
            for event in self.events.items():
                self.recomEvent(list(event))

        elif 0 == self.conf['doRecom']:
            self.events = {}

        #print('R:',self.events)

        #Dissociation
        if 1 == self.conf['doDisso']: # Switch for turning off dissociation 

            #for tag in self.boundCon.keys():
             #   self.checkDissoc(tag)
            self.Devents = self.getDissocEvs2()
            self.doDissocEvs(self.Devents)

        elif 0 == self.conf['doDisso']:
            self.Devents = []

        #print('D:',self.Devents)
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
            roll = disspB(np.random.uniform())
            if roll == None:
                continue
            res.append(roll)
        return res

    def getDissocEvs2(self):
        Fres = []
        Scon = {}
        for st in self.conf["StateList"]:
            tags = [tag for tag in self.boundCon.keys() if self.boundCon[tag].state == st]
            moms = np.array([self.boundCon[tag].mom4 for tag in self.boundCon.keys() if self.boundCon[tag].state == st ])
            vs = np.einsum('i,ij->ij', 1/moms[:,0], moms[:,1:]) 
            gams = 1/np.sqrt(1-np.einsum('ij,ij->i', vs, vs))
            Scon[st] = [tags, gams]
            #print('moms:',moms)
        # Distribution info
        
        for st in Scon.keys():
            RGAres = np.floor(self.rates['RGA'][st](Scon[st][1])*self.conf['dt']-np.random.rand(len(Scon[st][0]))).astype(int)+1
            # Sampled rate distribution
            self.recDump["RGArateSampDist"] = self.rates['RGA'][st](Scon[st][1]) 
            chresS = (RGAres,)
            chKey = ('RGA',)
            for i in range(len(RGAres)):
                resLine = np.array([res[i] for res in chresS]) 
                if np.all(resLine == 0):
                    continue
                else:
                    for m in range(len(resLine)):
                        if resLine[m] == 1:
                            Fres.append([chKey[m],Scon[st][0][i]])
                            break
        return Fres




    def doDissocEvs(self, evs):
        for result in evs:
            if result == None:
                continue
            else:
                #print('tag:',tag)
                #print('D:',result)
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

    def getRE_worker(self,targetList):
        for tg in targetList:
            self.getRecomEvents(tg[0],tg[1],tg[2])










    def getOccupations(self):
        return [len(self.boundCon.keys()),len(self.quarkCon.keys())]
    def getOccupationRatio(self):
        return len(self.boundCon.keys())/(self.conf['Nbb']+self.conf['NY'])
    def getMoms(self):
        return [np.linalg.norm(self.quarkCon[tag].mom) for tag in self.quarkCon.keys()]  
    def getMomsB(self, st):
        return [np.linalg.norm(self.boundCon[tag].mom) for tag in self.boundCon.keys() if self.boundCon[tag].state==st]
    def getMomDist(self):
        data = [np.norm(qrk.mom) for qrk in self.quarks]
        pBins = np.linspace(0,self.conf['pCut'],40)
        hist, bin_edges = np.histogram(data, bins = pBins)
        return hist
    def recLine(self):
        self.rec.append([self.time, self.getOccupationRatio()])

    def getNunbound(self):
        return len(self.quarkCon.keys())
    def getNbound(self, st):
        return sum([1 for tag in self.boundCon.keys() if self.boundCon[tag].state==st])

    def getNinXParts(self):
        tot = 0 
        for i in range(self.conf['NXPart']):
            for j in range(self.conf['NXPart']):
                for k in range(self.conf['NXPart']):
                    tot += len(self.XParts[i][j][k].keys())
        return tot
    
    def getNREvs(self):
        return len(self.events)
    def getNDEvs(self):

        return len(self.Devents)

    def measureRGRrateGam(self, st):
        qrkRates = []
        for i in range(self.conf['NXPart']):
            for j in range(self.conf['NXPart']):
                for k in range(self.conf['NXPart']):
                    for tagIn, ant in self.XParts[i][j][k].items(): # accessing 1 quark at a time at this level
                        if ant == 1:
                            continue
                        # Gets all  qrk.anti==0 qrks in partition ijk
                        #Iterates over each box in neighborhood and collects all antiquark tags
                        parTags = [tag for t in (i-1,i,i+1) for u in (j-1,j,j+1) for v in (k-1,k,k+1) for tag, ant in self.XParts[t % self.conf['NXPart']][u % self.conf['NXPart']][v % self.conf['NXPart']].items() if ant==1]
                        # For each tag in XPart[i][j][k] get all valid pairs of quark-antiquark momentums and separations
                        xpPairs = {tagIn+tagPar:[self.quarkCon[tagIn].mom4, self.quarkCon[tagPar].mom4, self.quarkCon[tagIn].pos, self.quarkCon[tagPar].pos] for tagPar in parTags}
                        # Then we need to boost all of these momentum pairs into p1 + p2

                         
                        #print('XPpairs: ', xpPairs)
                        pairtags = [pairtag for pairtag, xpP in xpPairs.items()]
                        
                        if len(pairtags) == 0:
                            return []
                        Xs = np.array([[xpP[2], xpP[3]] for pairtag, xpP in xpPairs.items()]) 
                        Ps = np.array([[xpP[0], xpP[1]] for pairtag, xpP in xpPairs.items()]) # [p1,p2]
                        
                        #print('BOOST MATS:',self.allBoost(self.HB(Xs[:][0], self.time)))
                        # Boost to hydro cell frame
                        RecPosns = (Xs[:,0]+Xs[:,1])/2 ### TELEPORTATION ISSUE HERE 
                        HBoosted = [ np.einsum('ijk,ik->ij',self.allBoost(self.HB(RecPosns, self.time)),Ps[:,0]), np.einsum('ijk,ik->ij',self.allBoost(self.HB(RecPosns, self.time)),Ps[:,1]) ]  # [(boosted) p1, p2] 4vec p1 and p2
                        #HBoosted = [Ps[:,0],Ps[:,1]]

                        #print('Xs',Xs)
                        #print('Ps',Ps)
                        #print('HBS',HBoosted[0])
                        # Boost to center of mass frame
                        Vcs = self.getVcell(HBoosted[0][:,1:]+HBoosted[1][:,1:])
                        #print('COMBO MEAL:',HBoosted[0][:,1:]+HBoosted[1][:,1:]  )
                        #print('Vcs:',Vcs)
                        CMBoosted = [np.einsum('ijk,ik->ij',self.allBoost(Vcs),HBoosted[0]), np.einsum('ijk,ik->ij',self.allBoost(Vcs),HBoosted[1])]
                        pRs = CMBoosted[0][:,1:]-CMBoosted[1][:,1:]
                        #print('pRsm:',np.min(np.einsum('ij,ij->i', pRs, pRs)))
                        #print('prs:', pRs)
                        #print('Xs:',Xs[:,0].shape)
                        #print('Xr',self.pDist(Xs[:][0],Xs[:][1])  )
                        inpVars = np.array([self.pDist(Xs[:,0],Xs[:,1]), np.einsum('ij,ij->i', pRs, pRs)])
                        #print('inpVars.shape:',inpVars.shape)

                        #inpVars = np.array([[np.linalg.norm(xpP[0][1:]-xpP[1][1:]),self.pDist(xpP[2], xpP[3])] for pairtag, xpP in xpPairs.items()])
                        #exit()
                        #print('inpVars',inpVars)
                        if np.size(inpVars) == 0:
                            #self.XPresCon[i][j][k] = []
                            return []


                        #print('hey',inpVars)
                        #print('cat',inpVars[:,0])
                        #print('thing',self.rates['RGR']['1S']((inpVars[:,0],inpVars[:,1])))
                        #print("RGRres", self.rates['RGR']['1S']((inpVars[:,0],inpVars[:,1]))*self.conf['dt']-np.random.rand(len(pairtags)))


                        # roll recombination in each channel
                        # floor(rateFunc(xr,pr)*dt - R) + 1  with (random number)
                        # RGR channel
                         
                        RGRres = 2*RGRsum2(inpVars[0],inpVars[1],np.linalg.norm(Vcs, axis=1),self.conf,st)
                        qrkRates.append([np.linalg.norm(self.quarkCon[tagIn].mom),np.sum(RGRres)])
                            
                     
            return np.array(qrkRates)
            
    
    def getRecomEventsMP(self,i,j,k):
        # Gets all  qrk.anti==0 qrks in partition ijk
        inTags = [tag for tag, ant in self.XParts[i][j][k].items() if ant==0]
        #Iterates over each box in neighborhood and collects all antiquark tags
        parTags = [tag for t in (i-1,i,i+1) for u in (j-1,j,j+1) for v in (k-1,k,k+1) for tag, ant in self.XParts[t % self.conf['NXPart']][u % self.conf['NXPart']][v % self.conf['NXPart']].items() if ant==1]
        # For each tag in XPart[i][j][k] get all valid pairs of quark-antiquark momentums and separations
        xpPairs = {tagIn+tagPar:[self.quarkCon[tagIn].mom4, self.quarkCon[tagPar].mom4, self.quarkCon[tagIn].pos, self.quarkCon[tagPar].pos] for tagIn in inTags for tagPar in parTags}
        # Then we need to boost all of these momentum pairs into p1 + p2

         
        #print('XPpairs: ', xpPairs)
        pairtags = [pairtag for pairtag, xpP in xpPairs.items()]
        
        if len(pairtags) == 0:
            return []
        Xs = np.array([[xpP[2], xpP[3]] for pairtag, xpP in xpPairs.items()]) 
        Ps = np.array([[xpP[0], xpP[1]] for pairtag, xpP in xpPairs.items()]) # [p1,p2]
        
        #print('BOOST MATS:',self.allBoost(self.HB(Xs[:][0], self.time)))
        # Boost to hydro cell frame
        RecPosns = (Xs[:,0]+Xs[:,1])/2 ### TELEPORTATION ISSUE HERE 
        HBoosted = [ np.einsum('ijk,ik->ij',self.allBoost(self.HB(RecPosns, self.time)),Ps[:,0]), np.einsum('ijk,ik->ij',self.allBoost(self.HB(RecPosns, self.time)),Ps[:,1]) ]  # [(boosted) p1, p2] 4vec p1 and p2
        #HBoosted = [Ps[:,0],Ps[:,1]]

        #print('Xs',Xs)
        #print('Ps',Ps)
        #print('HBS',HBoosted[0])
        # Boost to center of mass frame
        Vcs = self.getVcell(HBoosted[0][:,1:]+HBoosted[1][:,1:])
        #print('COMBO MEAL:',HBoosted[0][:,1:]+HBoosted[1][:,1:]  )
        #print('Vcs:',Vcs)
        CMBoosted = [np.einsum('ijk,ik->ij',self.allBoost(Vcs),HBoosted[0]), np.einsum('ijk,ik->ij',self.allBoost(Vcs),HBoosted[1])]
        pRs = CMBoosted[0][:,1:]-CMBoosted[1][:,1:]
        #print('pRsm:',np.min(np.einsum('ij,ij->i', pRs, pRs)))
        #print('prs:', pRs)
        #print('Xs:',Xs[:,0].shape)
        #print('Xr',self.pDist(Xs[:][0],Xs[:][1])  )
        inpVars = np.array([self.pDist(Xs[:,0],Xs[:,1]), np.einsum('ij,ij->i', pRs, pRs)])
        #print('inpVars.shape:',inpVars.shape)

        #inpVars = np.array([[np.linalg.norm(xpP[0][1:]-xpP[1][1:]),self.pDist(xpP[2], xpP[3])] for pairtag, xpP in xpPairs.items()])
        #exit()
        #print('inpVars',inpVars)
        if np.size(inpVars) == 0:
            #self.XPresCon[i][j][k] = []
            return []


        #print('hey',inpVars)
        #print('cat',inpVars[:,0])
        #print('thing',self.rates['RGR']['1S']((inpVars[:,0],inpVars[:,1])))
        #print("RGRres", self.rates['RGR']['1S']((inpVars[:,0],inpVars[:,1]))*self.conf['dt']-np.random.rand(len(pairtags)))


        # roll recombination in each channel
        # floor(rateFunc(xr,pr)*dt - R) + 1  with (random number)R in (0,1) <-- this should be 1 for a recombination and 0 otherwise
        # RGR channel

        RGRres = {st:np.floor(2*RGRsum2(inpVars[0],inpVars[1],np.linalg.norm(Vcs, axis=1),self.conf,st)*self.conf['dt']-np.random.rand(len(pairtags))).astype(int)+1 for st in self.conf['StateList']}
        

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
        return res  #This is a list with elements ['name1name2',[state, channel]]   
    
    def checkREvents(self, evIn):
        events = {}
        ColCheck = {}
        for ev in evIn: 
            if ev[0][:8] not in ColCheck.keys() and ev[0][8:] not in ColCheck.values():
                ColCheck[ev[0][:8]] = ev[0][8:]
                events[ev[0]] = ev[1] # add to event list
            else:
                
                continue
                # This is where conflicts will be handled but at first pass they will just be dropped !!!!!!!!!!!!!!!!!!! (This is roughly equivalent to how it worked 
                # before with these just never getting rolled in the first place)
        return events
        #return [[tag,events[tag]] for tag in events.keys()]

   # min distance in the periodic box between two points
    def pDist(self, x1, x2):
        delta = np.abs(x1 - x2)
        periodic_delta = np.minimum(delta, self.conf['L'] - delta)
        return np.linalg.norm(periodic_delta, axis=1)
      
    


    def getVcell(self, pcm):
        return pcm/np.sqrt(((2*self.conf['Mb'])**2) + np.einsum('ij,ij->i',pcm,pcm))[:,np.newaxis]
 
    
    def allBoost(self, v): # v is a vector of velocity vectors vi 
        vx, vy, vz = v.T # get the x y and z components of each vi
        vM = np.linalg.norm(v, axis=1) # get vector of magnitudes of each velocity
        g = 1/np.sqrt(1 - (vM**2)) # gamma
        vvT = np.einsum('ij,ik->ijk',v,v) # batched outer product of v and velocity
        res = np.zeros((len(v),4,4)) # init containers
        res[:, 0, 0] = g # 00 elem set to gamma
        #print(v.shape)
        res[:, 0, 1:] = -g[:,np.newaxis]*v # 0j set to -g*v
        res[:, 1:, 0] = -g[:,np.newaxis]*v # i0 to -g*v 
        #print('v:',v)
        #print(len(v))
        res[:, 1:, 1:] = np.tile(np.eye(3), (len(v), 1, 1)) + (g[:,np.newaxis,np.newaxis]-np.ones((len(v),3,3)))*vvT/vM[:,np.newaxis,np.newaxis] # i-123 j-123 set to 1 - (g-1)vvT/vM
        return res
    
    
    
    
    
    
    
    
    
    
def regenUpdate_worker(queue, box, targetList, Rs, thN):
    try:
        np.random.seed(Rs)
        results = []
        for tg in targetList: # for each target cell in this process's targetList get the events in that cell
            #print('Thread:',thN,' target:',tg)
            ret = box.getRecomEventsMP(tg[0],tg[1],tg[2])
            for res in ret: # then for each event in that cell, add it to the result queue
                #results.append(res)
                queue.put(res)
        #print(thN,' All fetched')
        return
        #multiprocessing.current_process().terminate()
    except Exception as e:
        print(colr('Error',(200,0,0)),'in',colr('Thread'+str(thN),(150,150,50)),':',f"{e}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
