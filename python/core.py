# Core

import numpy as np


# mathFunc2
from mathFunc2 import sampleSphere
from mathFunc2 import mom324
from mathFunc2 import chooseStateInit
from mathFunc2 import BoostL
from mathFunc2 import Vcell
from mathFunc2 import pDist
from mathFunc2 import RGRsumC
from mathFunc2 import IRQsumC
from mathFunc2 import IRGsumC
from mathFunc2 import vFp
from mathFunc2 import vFp4, getIM, flattenL

# momentumSampling
from momentumSampling import doRGAdiss

from momentumSampling import doRGRrecom

from momentumSampling import recThermal
from momentumSampling import disThermal

# config
from config import stepClock
from config import colr

# hydro
from hydro import hydroBac

import threading
import time



import multiprocessing



# Heavy quark
class quark:
    def __init__(self, conf, anti=0, mom=None, pos=None, name=None):
        self.conf = conf
        self.anti = anti
        self.name = hex8() if name == None else name
        self.pos = np.random.rand(3)*conf['L'] if pos == None else pos
        if type(mom) == type(None):
            if 0 == conf['pSampleType']:
                self.mom = mom324(conf['fBinvCDF_b'](np.random.rand())*sampleSphere(1)[0], self.conf['Mb'])
        else:
            self.mom = mom
        self.newposT()
    def Xstep(self):
        self.pos = (self.pos + (self.mom[1:]/np.sqrt((self.conf['Mb']**2)+np.dot(self.mom[1:],self.mom[1:])))*self.conf['dt'])%self.conf['L']
    def Pstep(self):
        return None
    def posT(self):
        newXP = np.floor(self.pos*self.conf['NXPart']/self.conf['L']).astype(int) 
        if np.all(newXP == self.XPart):
            return False, None, None
        else:
            oldXP = self.XPart
            self.XPart = newXP
            return True, oldXP, newXP
    def newposT(self):
        self.XPart = np.floor(self.pos*self.conf['NXPart']/self.conf['L']).astype(int) 
    def getnewposT(self):
        self.XPart = np.floor(self.pos*self.conf['NXPart']/self.conf['L']).astype(int)
        return self.XPart
    def getposT(self):
        return self.XPart 

# Bound state
class bound:
    def __init__(self, conf, st, quarks=None, mom=None, pos=None):
        self.conf = conf
        self.st = st
        if quarks == None:
            q1 = quark(self.conf, anti=0, mom=mom324(np.array([0.,0.,0.]), conf['Mb']))
            q2 = quark(self.conf, anti=1, mom=mom324(np.array([0.,0.,0.]), conf['Mb']))
            self.quarks = [q1,q2]
            if 0 == conf['pSampleType']: #Thermal sampling
                self.mom = mom324(conf['fBinvCDF_'+self.st](np.random.rand())*sampleSphere(1)[0], self.conf['M'+self.st])
            self.pos = np.random.rand(3)*conf['L']  
        else:
            self.quarks = quarks
            self.mom = mom
            self.pos = (quarks[0].pos+quarks[1].pos)/2 #Position set to center of mass
        self.name = self.quarks[0].name + self.quarks[1].name
    def Xstep(self):
        self.pos = (self.pos + (self.mom[1:]/np.sqrt((self.conf['Mb']**2)+np.dot(self.mom[1:],self.mom[1:])))*self.conf['dt'])%self.conf['L']
    def Pstep(self):
        return None
    def dissociate(self, pQ, pQ_): # When this function is called it  returns the constituent quarks as [quark, anti]
        #Set constituent quark positions to the position of dissociated bound
        for qrk in self.quarks:
            qrk.pos = self.pos 
        self.quarks[0].mom4 = pQ
        self.quarks[1].mom4 = pQ_
        return self.quarks


class box:
    def __init__(self, conf):
        self.conf = conf
        self.cl = stepClock(self.conf['tFn'])
        self.ti = 0
        self.time = 0.0

        self.HB = hydroBac(conf)
        self.rec = []

        quarkT = [quark(self.conf, anti=a) for i in range(conf['Nbb']) for a in (0,1)]
        self.quarkC = {qrk.name:qrk for qrk in quarkT}
        boundT = [bound(self.conf, st=chooseStateInit(self.conf)) for i in range(conf['NY'])]
        self.boundC = {bnd.name:bnd for bnd in boundT} 

        self.initXPart() 

        #self.XPresCon = [[[[] for i in range(self.conf['NXPart'])] for j in range(self.conf['NXPart'])] for k in range(self.conf['NXPart'])] # These are the containers in which the multithreaded recombination puts results

        self.initDump()

    def step(self):
        #Start step timer
        self.cl.start()
        
        self.doXstep()
        self.doPstep()

        # Update space partitions
        self.updateXPart()

        # Event sampling
        # Regeneration
        if 1 == self.conf['doRecom']:
            self.recomEvents = self.getRecomEvents()
        elif 0 == self.conf['doRecom']:
            self.recomEvents = {}
        
        #Dissociation
        if 1 == self.conf['doDisso']:
            self.dissoEvents = self.getDissoEvents()
        elif 0 == self.conf['doDisso']:
            self.dissoEvents = {}

        #print('DEVENTS:',self.dissoEvents)
        #print('REVENTS:',self.recomEvents)

        # Momentum sampling
        # Regeneration
        if 1 == self.conf['doRecom']:
            self.doRecomEvents(self.recomEvents)
        #Dissociation
        if 1 == self.conf['doDisso']:
            self.doDissoEvents(self.dissoEvents)

        # Increment time
        self.time += self.conf['dt']
        self.ti += 1 
        # Stop timer
        self.cl.stop()






    #X and P step functions
    def doXstep(self):
        for tag in self.quarkC.keys():
            self.quarkC[tag].Xstep()
        for tag in self.boundC.keys():
            self.boundC[tag].Xstep()
    def doPstep(self):
        for tag in self.quarkC.keys():
            self.quarkC[tag].Pstep()
        for tag in self.boundC.keys():
            self.boundC[tag].Pstep()

    # Space partition functions
    def initXPart(self):
        self.XParts = [[[{} for i in range(self.conf['NXPart'])] for j in range(self.conf['NXPart'])] for k in range(self.conf['NXPart'])]
        for tag, qrk in self.quarkC.items():
            XPt = np.floor(qrk.pos*self.conf['NXPart']/self.conf['L']).astype(int)
            self.XParts[XPt[0]][XPt[1]][XPt[2]][tag] = qrk.anti
    def updateXPart(self):
        for tag, qrk in self.quarkC.items():
            XPmove, old, new = qrk.posT()
            if XPmove:
                del self.XParts[old[0]][old[1]][old[2]][tag]
                self.XParts[new[0]][new[1]][new[2]][tag] = qrk.anti

    # Dissociation event sampling
    def getDissoEvents(self):
        Dres = {}
        for st in self.conf['StateList']:
            tags = [tag for tag in self.boundC.keys() if self.boundC[tag].st == st]
            moms = np.array([self.boundC[tag].mom for tag in self.boundC.keys() if self.boundC[tag].st == st ])
            vs = vFp4(moms)
            for ch in self.conf['ChannelList']:
                if ch not in ('RGA','IDQ','IDG'):
                    continue
                rs = np.floor(self.conf[ch+'_rate'+st](vs)*self.conf['dt']-np.random.rand(len(vs))).astype(int)+1
                Dres.update({tags[i]:ch for i in np.nonzero(rs)[0]}) ### !!!!!
        return list(Dres.items())
    def doDissoEvents(self, evs):
        #print('doing events ',evs)
        for r in evs:
            if r[1] == 'RGA':
                #pQ, pQ_ = self.doRGAdiss(result[1])
                pQ, pQ_ = disThermal(self.conf)
                self.dissociateBoundTag(r[0], pQ, pQ_)
                continue
            elif r[1] == 'IDQ':
                #Do x channel
                pQ, pQ_ = disThermal(self.conf)
                self.dissociateBoundTag(r[0], pQ, pQ_)
                continue
            elif r[1] == 'IDG':
                #Do y channel
                pQ, pQ_ = disThermal(self.conf)
                self.dissociateBoundTag(r[0], pQ, pQ_)
                continue    
    def dissociateBoundTag(self, tag, pQ, pQ_):
        #print('DBG0:',len(self.boundCon.keys()))
        qrks = self.boundC[tag].dissociate(pQ, pQ_)
        #print('DBG1:',len(self.boundCon.keys()))
        del self.boundC[tag]
        #print('DBG2:',len(self.boundCon.keys()))
        for qrk in qrks:
            self.quarkC[qrk.name] = qrk
            loc = qrk.getnewposT()
            self.XParts[loc[0]][loc[1]][loc[2]][qrk.name] = qrk.anti 


    # Recombination event sampling
    def getRecomEvents(self):
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
        return self.checkREvents(eventsRaw)
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
    def doRecomEvents(self, evs): # these events come as ['name1name2',[state, channel]]
        for ev in evs.items():
            evL = list(ev)
            if ev[1][1] == 'RGR':
                #kL = self.doRGRrecom(ev[0][:8], ev[0][8:], ev[1][0])
                kL = recThermal(ev[0][:8], ev[0][8:], ev[1][0], self.conf)
            elif ev[1][1] == 'IRQ':
                #kL = self.doIRQrecom(ev[0][:8], ev[0][8:], ev[1][0])
                kL = recThermal(ev[0][:8], ev[0][8:], ev[1][0], self.conf)
            elif ev[1][1] == 'IRG':
                #kL = self.doIRGrecom(ev[0][:8], ev[0][8:], ev[1][0])
                kL = recThermal(ev[0][:8], ev[0][8:], ev[1][0], self.conf)
            self.combineTags(ev[0], ev[1][0], kL)

    def combineTags(self, tags, state, qP):
        # quarks need to be removed from XParts
        loc1 = self.quarkC[tags[:8]].XPart
        del self.XParts[loc1[0]][loc1[1]][loc1[2]][tags[:8]]
        loc2 = self.quarkC[tags[8:]].XPart
        del self.XParts[loc2[0]][loc2[1]][loc2[2]][tags[8:]]

        # create new bound object and pop quarks out of quarkD, and add bound to bound list
        self.boundC[tags] = bound(self.conf, quarks=[self.quarkC.pop(tags[:8]),self.quarkC.pop(tags[8:])], st=state, mom=qP)    

    def getRecomEventsMP(self,i,j,k):
        # Gets all  qrk.anti==0 qrks in partition ijk
        inTags = [tag for tag, ant in self.XParts[i][j][k].items() if ant==0]
        #Iterates over each box in neighborhood and collects all antiquark tags
        parTags = [tag for t in (i-1,i,i+1) for u in (j-1,j,j+1) for v in (k-1,k,k+1) for tag, ant in self.XParts[t % self.conf['NXPart']][u % self.conf['NXPart']][v % self.conf['NXPart']].items() if ant==1]
        # For each tag in XPart[i][j][k] get all valid pairs of quark-antiquark momentums and separations
        xpPairs = {tagIn+tagPar:[self.quarkC[tagIn].mom, self.quarkC[tagPar].mom, self.quarkC[tagIn].pos, self.quarkC[tagPar].pos] for tagIn in inTags for tagPar in parTags}
        # Then we need to boost all of these momentum pairs into p1 + p2
        pairtags = [pairtag for pairtag, xpP in xpPairs.items()]
        if len(pairtags) == 0:
            return []
        Xs = np.array([[xpP[2], xpP[3]] for pairtag, xpP in xpPairs.items()]) 
        Ps = np.array([[xpP[0], xpP[1]] for pairtag, xpP in xpPairs.items()]) # [p1,p2]
        # Boost to hydro cell frame
        RecPosns = (Xs[:,0]+Xs[:,1])/2 ### TELEPORTATION ISSUE HERE 
        HBoosted = [ np.einsum('ijk,ik->ij',BoostL(self.HB(RecPosns, self.time)),Ps[:,0]), np.einsum('ijk,ik->ij',BoostL(self.HB(RecPosns, self.time)),Ps[:,1]) ]  # [(boosted) p1, p2] 4vec p1 and p2
        # Boost to center of mass frame
        Vcs = Vcell(HBoosted[0]+HBoosted[1])
        CMBoosted = [np.einsum('ijk,ik->ij',BoostL(Vcs),HBoosted[0]), np.einsum('ijk,ik->ij',BoostL(Vcs),HBoosted[1])]
        pRs = (1/2)*(CMBoosted[0][:,1:]-CMBoosted[1][:,1:])
        inpVars = np.array([pDist(self.conf['L'],Xs[:,0],Xs[:,1]), np.sqrt(np.einsum('ij,ij->i', pRs, pRs))])
        if np.size(inpVars) == 0:
            return []

        # roll recombination in each channel
        # floor(rateFunc(xr,pr)*dt - R) + 1  with (random number)R in (0,1) <-- this should be 1 for a recombination and 0 otherwise
        

        Rres = {}
        for ch in self.conf['ChannelList']:
            if ch not in ('RGR','IRQ','IRG'):
                continue
            for st in self.conf['StateList']:
                if ch == 'RGR':
                    rs = np.floor(RGRsumC(inpVars[0],np.linalg.norm(Vcs, axis=1),inpVars[1],self.conf,st)*self.conf['dt']-np.random.rand(len(pairtags))).astype(int)+1
                elif ch == 'IRQ':
                    rs = np.floor(IRQsumC(inpVars[0],np.linalg.norm(Vcs, axis=1),inpVars[1],self.conf,st)*self.conf['dt']-np.random.rand(len(pairtags))).astype(int)+1
                elif ch == 'IRG':
                    rs = np.floor(IRGsumC(inpVars[0],np.linalg.norm(Vcs, axis=1),inpVars[1],self.conf,st)*self.conf['dt']-np.random.rand(len(pairtags))).astype(int)+1

    
                for ind in np.nonzero(rs)[0]:
                    Rres.update({pairtags[ind]:[st,ch]})
        
        return [[tag, ev] for tag, ev in Rres.items()]


    def initDump(self):
        self.recDump = {}
        #self.recDump['RGRrateP1S'] = self.measureRGRrateGam('1S')
        self.recDump['RGArateSampDist'] = []
        self.recDump['qEin'] = [0 for i in range(self.conf['tFn'])]
        self.recDump['qEout'] = [0 for i in range(self.conf['tFn'])]
        self.recDump['evLog'] = []

    def recLine(self):
        self.rec.append([self.time, self.getOccupationRatio()])
    def getOccupationRatio(self):
        return len(self.boundC.keys())/(self.conf['Nbb']+self.conf['NY'])
    def getEb(self): # returns sum(p[0] for each b and bbar)/Nb(t)
        return np.sum(np.array([self.quarkC[tag].mom[0] for tag in self.quarkC.keys()]))/(len(self.quarkC.keys()))
    def getEY(self): # returns sum(p[0] for each bound)/NY(t) 
        return np.sum(np.array([self.boundC[tag].mom[0] for tag in self.boundC.keys()]))/(len(self.boundC.keys()))
    def getEMix(self): # = getTot4p()[0]
        return (( np.sum(np.array([self.quarkC[tag].mom for tag in self.quarkC.keys()]), axis=0) + np.sum(np.array([self.boundC[tag].mom for tag in self.boundC.keys()]), axis=0) ) / ((self.conf['Nbb']*2)+(self.conf['NY']*2)))[0]

    def getMYs(self): # returns avg invariant mass of bounds (debug)
        return np.sum(np.array([getIM(self.boundC[tag].mom) for tag in self.boundC.keys()]))/len(self.boundC.keys())

    def getMbs(self): # returns avg invariant mass of b and bbars (debug)
        return np.sum(np.array([getIM(self.quarkC[tag].mom) for tag in self.quarkC.keys()]))/len(self.quarkC.keys())

    def getNREvs(self):
        return len(self.recomEvents)
    def getNDEvs(self):
        print('ND2:',self.dissoEvents)
        return len(self.dissoEvents)

    def measureDrate(self):
        DCh = list(set(self.conf['ChannelList']) & set(['RGA','IDQ','IDG'])) 
        Dres = {ch:{st:[] for st in self.conf['StateList']} for ch in DCh}
        for st in self.conf['StateList']:
            tags = [tag for tag in self.boundC.keys() if self.boundC[tag].st == st]
            moms = np.array([self.boundC[tag].mom for tag in self.boundC.keys() if self.boundC[tag].st == st ])
            vs = vFp4(moms)
            for ch in self.conf['ChannelList']:
                if ch not in ('RGA','IDQ','IDG'):
                    continue
                rs = self.conf[ch+'_rate'+st](vs)
                Dres[ch][st].append(rs)
        return Dres

    def measureRrate(self):
        RCh = list(set(self.conf['ChannelList']) & set(['RGR','IRQ','IRG']))
        Rres = {ch:{st:[] for st in self.conf['StateList']} for ch in RCh}
        for i in range(self.conf['NXPart']):
            for j in range(self.conf['NXPart']):
                for k in range(self.conf['NXPart']):
                    if (i+j+k)%self.conf['RGRrateOpt'] != 0:
                        continue
                    for tagIn, ant in self.XParts[i][j][k].items(): # accessing 1 quark at a time at this level
                        if ant == 1:
                            continue
                        # Gets all  qrk.anti==0 qrks in partition ijk
                        #Iterates over each box in neighborhood and collects all antiquark tags
                        parTags = [tag for t in (i-1,i,i+1) for u in (j-1,j,j+1) for v in (k-1,k,k+1) for tag, ant in self.XParts[t % self.conf['NXPart']][u % self.conf['NXPart']][v % self.conf['NXPart']].items() if ant==1]
                        # For each tag in XPart[i][j][k] get all valid pairs of quark-antiquark momentums and separations
                        xpPairs = {tagIn+tagPar:[self.quarkC[tagIn].mom, self.quarkC[tagPar].mom, self.quarkC[tagIn].pos, self.quarkC[tagPar].pos] for tagPar in parTags}
                        # Then we need to boost all of these momentum pairs into p1 + p2

                        pairtags = [pairtag for pairtag, xpP in xpPairs.items()]
                        
                        if len(pairtags) == 0:
                            continue

                        Xs = np.array([[xpP[2], xpP[3]] for pairtag, xpP in xpPairs.items()]) 
                        Ps = np.array([[xpP[0], xpP[1]] for pairtag, xpP in xpPairs.items()]) # [p1,p2]
                        
                        # Boost to hydro cell frame
                        RecPosns = (Xs[:,0]+Xs[:,1])/2 ### TELEPORTATION ISSUE HERE 
                        HBoosted = [ np.einsum('ijk,ik->ij',BoostL(self.HB(RecPosns, self.time)),Ps[:,0]), np.einsum('ijk,ik->ij',BoostL(self.HB(RecPosns, self.time)),Ps[:,1]) ]  # [(boosted) p1, p2] 4vec p1 and p2
                        #HBoosted = [Ps[:,0],Ps[:,1]]

                        # Boost to center of mass frame
                        Vcs = Vcell(HBoosted[0]+HBoosted[1])
                        #print('#########################',Vcs.shape)
                        CMBoosted = [np.einsum('ijk,ik->ij',BoostL(Vcs),HBoosted[0]), np.einsum('ijk,ik->ij',BoostL(Vcs),HBoosted[1])]
                        pRs = (1/2)*(CMBoosted[0][:,1:]-CMBoosted[1][:,1:])
                        inpVars = np.array([pDist(self.conf['L'],Xs[:,0],Xs[:,1]), np.sqrt(np.einsum('ij,ij->i', pRs, pRs))])
                        if np.size(inpVars) == 0:
                            #self.XPresCon[i][j][k] = []
                            continue


                        
                        for ch in RCh:
                            for st in self.conf['StateList']:
                                if ch == 'RGR':
                                    rs = RGRsumC(inpVars[0],np.linalg.norm(Vcs, axis=1),inpVars[1],self.conf,st)
                                elif ch == 'IRQ':
                                    rs = IRQsumC(inpVars[0],np.linalg.norm(Vcs, axis=1),inpVars[1],self.conf,st)
                                elif ch == 'IRG':
                                    rs = IRGsumC(inpVars[0],np.linalg.norm(Vcs, axis=1),inpVars[1],self.conf,st)
                                Rres[ch][st].append(np.sum(rs))
        return Rres
    def measureRrateMP(self,i,j,k):
        RCh = list(set(self.conf['ChannelList']) & set(['RGR','IRQ','IRG']))
        Rres = {ch:{st:[] for st in self.conf['StateList']} for ch in RCh}
        #print('HEllo')
        for tagIn, ant in self.XParts[i][j][k].items(): # accessing 1 quark at a time at this level
            if ant == 1:
                continue
            # Gets all  qrk.anti==0 qrks in partition ijk
            #Iterates over each box in neighborhood and collects all antiquark tags
            parTags = [tag for t in (i-1,i,i+1) for u in (j-1,j,j+1) for v in (k-1,k,k+1) for tag, ant in self.XParts[t % self.conf['NXPart']][u % self.conf['NXPart']][v % self.conf['NXPart']].items() if ant==1]
            # For each tag in XPart[i][j][k] get all valid pairs of quark-antiquark momentums and separations
            xpPairs = {tagIn+tagPar:[self.quarkC[tagIn].mom, self.quarkC[tagPar].mom, self.quarkC[tagIn].pos, self.quarkC[tagPar].pos] for tagPar in parTags}
            # Then we need to boost all of these momentum pairs into p1 + p2

            pairtags = [pairtag for pairtag, xpP in xpPairs.items()]
            
            if len(pairtags) == 0:
                for ch in RCh:
                    for st in self.conf['StateList']:
                        Rres[ch][st].append(0)
                continue

            Xs = np.array([[xpP[2], xpP[3]] for pairtag, xpP in xpPairs.items()]) 
            Ps = np.array([[xpP[0], xpP[1]] for pairtag, xpP in xpPairs.items()]) # [p1,p2]
            
            # Boost to hydro cell frame
            RecPosns = (Xs[:,0]+Xs[:,1])/2 ### TELEPORTATION ISSUE HERE 
            HBoosted = [ np.einsum('ijk,ik->ij',BoostL(self.HB(RecPosns, self.time)),Ps[:,0]), np.einsum('ijk,ik->ij',BoostL(self.HB(RecPosns, self.time)),Ps[:,1]) ]  # [(boosted) p1, p2] 4vec p1 and p2
            #HBoosted = [Ps[:,0],Ps[:,1]]

            # Boost to center of mass frame
            Vcs = Vcell(HBoosted[0]+HBoosted[1])
            #print('#########################',Vcs.shape)
            CMBoosted = [np.einsum('ijk,ik->ij',BoostL(Vcs),HBoosted[0]), np.einsum('ijk,ik->ij',BoostL(Vcs),HBoosted[1])]
            pRs = (1/2)*(CMBoosted[0][:,1:]-CMBoosted[1][:,1:])
            inpVars = np.array([pDist(self.conf['L'],Xs[:,0],Xs[:,1]), np.sqrt(np.einsum('ij,ij->i', pRs, pRs))])
            if np.size(inpVars) == 0:
                #self.XPresCon[i][j][k] = []
                continue


           
            for ch in RCh:
                for st in self.conf['StateList']:
                    if ch == 'RGR':
                        rs = RGRsumC(inpVars[0],np.linalg.norm(Vcs, axis=1),inpVars[1],self.conf,st)
                    elif ch == 'IRQ':
                        rs = IRQsumC(inpVars[0],np.linalg.norm(Vcs, axis=1),inpVars[1],self.conf,st)
                    elif ch == 'IRG':
                        rs = IRGsumC(inpVars[0],np.linalg.norm(Vcs, axis=1),inpVars[1],self.conf,st)
                    Rres[ch][st].append(np.sum(rs))
        return Rres
    def measureRrateMPH(self):
        targetList = [[] for i in range(self.conf['NThreads'])]
        for i in range(self.conf['NXPart']):
            for j in range(self.conf['NXPart']):
                for k in range(self.conf['NXPart']):
                    targetList[(i+(j*self.conf['NXPart'])+(k*(self.conf['NXPart']**2)))%self.conf['NThreads']].append([i,j,k])
            
        rateRaw = []
        result_queue = multiprocessing.Queue()
        processList = []
        for i in range(self.conf['NThreads']):
            process = multiprocessing.Process(target=regenRate_worker, args=(result_queue, self, targetList[i], np.random.randint(2**32), i))
            processList.append(process)
            process.start()

        while any(process.is_alive() for process in processList) or not result_queue.empty():
            #if len(processList)!=0:
                #print('remaining:',processList)
            while not result_queue.empty():
                rateRaw.append(result_queue.get())
        #print('RR: ',rateRaw)
        for process in processList:
            process.join()

        RCh = list(set(self.conf['ChannelList']) & set(['RGR','IRQ','IRG']))
        #res = {ch:{st:[] for st in self.conf['StateList']} for ch in RCh}
        #stitch togetther partial results
        res = {ch:{st:flattenL([it[ch][st] for it in rateRaw]) for st in self.conf['StateList']} for ch in RCh}
        # Get events and manage conflicts
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

def regenRate_worker(queue, box, targetList, Rs, thN):
    try:
        np.random.seed(Rs)
        results = []
        for tg in targetList: # for each target cell in this process's targetList get the events in that cell
            #print('Thread:',thN,' target:',tg)
            ret = box.measureRrateMP(tg[0],tg[1],tg[2])

            #print('ret: ',ret)
            queue.put(ret)
        #print(thN,' All fetched')
        return
        #multiprocessing.current_process().terminate()
    except Exception as e:
        print(colr('Error',(200,0,0)),'in',colr('Thread'+str(thN),(150,150,50)),':',f"{e}")


# Generates a random 8 character hex string
def hex8():
    return '%08x' % np.random.randint(16**8)

