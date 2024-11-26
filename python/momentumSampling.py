import numpy as np

from mathFunc2 import sampleSphere
from mathFunc2 import mom324

### Dissociation

# RGA
def doRGAdiss(bnd):
    #Collect dynamic variables

    pB = self.bnd.mom
    vV = pB[1:]/pB[0]
    v = np.linalg.norm(vV)
    gam = 1/np.sqrt(1-np.power(v,2))
    #Sample gluon momentum magnitude
    resamp = True
    st = self.bnd.st
    while resamp:
        qtry = np.random.uniform(conf['E'+st],conf['E'+st]*conf['ECut']) 
        if np.random.uniform() < self.dists['RGA'][st](np.array([gam, qtry])):
            resamp = False
    #qmag = (qtry - self.conf['E'+st])/self.conf['M'+st]
    qmag = qtry
    #Solve for CosTheta = x of gluon momentum
    r = np.random.random()
    B = gam*qmag/self.conf['T']
    C = (r*np.log(1-np.exp(-B*(1+v))))+((1-r)*np.log(1-np.exp(-B*(1-v))))
    CosTheta = (1/v)*(-1-((1/B)*np.log(1-np.exp(C))))
    SinTheta = np.sqrt(1-np.power(CosTheta,2)) # 1 = sin^2 + cos^2
    #Set gluon momentum
    qg = np.array([qmag, qmag*SinTheta, 0, qmag*CosTheta])

    #Solve relative momentum part
    #prelM = np.sqrt(self.conf['Mb']*(qmag - self.conf['E'+st]))
    prelM = prFqRG(self.conf, qmag, st)
    CosThetaRel = (np.random.random()*2)-1 # [-1,1]
    SinThetaRel = np.sqrt(1-np.power(CosThetaRel,2)) # 1 = sin^2 + cos^2
    PhiRel = np.random.random()*np.pi*2 # [0,2Pi]
    prel = np.array([prelM*SinThetaRel*np.cos(PhiRel), prelM*SinThetaRel*np.sin(PhiRel), prelM*CosThetaRel])
    #Set and rotate pQ and pQ_
    RotMat = self.getRotMat(pB)
    pQ = RotMat @ (prel + (qg[1:]/2))
    pQ_ = RotMat @ (-prel + (qg[1:]/2))
    #print('ShapeCheck: ', self.HB(np.array([self.boundCon[tar].pos]), self.time).shape)
    Bs = self.allBoost(np.array([-self.HB(np.array([self.boundCon[tar].pos]), self.time)[0], -vV])) # gets both vcell boost and hydro boost
    self.recDump['qEin'][self.ti] += (Bs[0] @ (Bs[1] @ (np.insert(RotMat @ qg[1:], 0, qmag))))[0] # Energy of q in 
    pQL, pQ_L, = Bs[0] @ (Bs[1] @ np.insert(pQ, 0, np.sqrt((pQ @ pQ) + np.power(self.conf['Mb'],2)))), Bs[0] @ (Bs[1] @ np.insert(pQ_, 0, np.sqrt((pQ_ @ pQ_) + np.power(self.conf['Mb'],2)))) 
    #print('Dchk:', (pB[0]+qmag)-(pQL[0]+pQ_L[0]))
    #print('|pQ|', np.linalg.norm(pQ_L[1:]))
    return pQL, pQ_L
    # returns rotated and boosted pQ, pQ_


### Recombination

#RGR
def doRGRrecom(self, n1, n2, st): # RGR momentum sampling
    x = self.pDist(np.array([self.quarkCon[n1].pos]), np.array([self.quarkCon[n2].pos]))[0] # x distance
    xRec = (self.quarkCon[n1].pos + self.quarkCon[n2].pos)/2 # recombination position !!!TELEPORTATION ISSUE
    p1, p2 = self.quarkCon[n1].mom4, self.quarkCon[n2].mom4 # quark momentums
    HBM = self.allBoost(self.HB(np.array([xRec]), self.time))[0] # get Hydro boost matrix
    Hp1, Hp2 = HBM @ p1, HBM @ p2 # do hydro boost
    Vc = self.getVcell2(np.array([Hp1+Hp2]))[0] # get Vcell
    VcM = np.linalg.norm(Vc) # scalar vcell
    g = 1/np.sqrt(1-np.power(VcM,2)) # gamma
    CBM = self.allBoost(np.array([Vc]))[0] # get vcell boost
    CHp1, CHp2 = CBM @ Hp1, CBM @ Hp2 # do vcell boosts
    prel = (CHp1[1:] - CHp2[1:])/2 # relative momentum 3-vec
    #q = ((prel @ prel)/self.conf['Mb']) + self.conf['E'+st] # gluon momentum fixed
    q = qFprRG(self.conf, np.linalg.norm(prel), st)
    # Sample CosThetag
    resamp = True
    while resamp:
        CosThetag = (np.random.random()*2)-1 # c on [-1,1]
        r = np.random.random() # r on [0,1]
        if r*(1/(np.exp(g*(1+VcM)*q/self.conf['T'])-1)) < r*(1/(np.exp(g*(1+(VcM*CosThetag))*q/self.conf['T'])-1)): #Check condition
            resamp = False
    SinThetag = np.sqrt(1-np.power(CosThetag,2)) # 1 = sin^2 + cos^2
    Phig = np.random.random()*np.pi*2 # Phig on [0,2Pi]
    qV = np.array([q*SinThetag*np.cos(Phig), q*SinThetag*np.sin(Phig), q*CosThetag]) # gluon momentum 3-vec
    RM = self.getRotMat(Vc) # Rotation matrix associated with Vc
    kRot = RM @ (-qV) # Apply RM to -qV = krest
    RB = self.allBoost(np.array([-self.HB(np.array([xRec]), self.time)[0], -Vc])) # get reverse boosts [-vHydro, -vcell]
    klab = RB[0] @ (RB[1] @ np.insert(kRot, 0, np.sqrt(np.power(self.conf['M'+st],2) + (kRot @ kRot)))) # apply reverse boosts to kRot 4-vec
    self.recDump['qEout'][self.ti] += (RB[0] @ (RB[1] @ np.insert(-kRot, 0, q)))[0] # Energy of q in lab frame  

    kE = np.sqrt(np.power(self.conf['M'+st],2)+(kRot @ kRot))
    #print('CHK:',(kE+q)-(CHp1[0]+CHp2[0]))
    #print('C3:', np.linalg.norm(CHp1[1:] + CHp2[1:]))
    #print('C4:', q - np.sqrt(kRot @ kRot))
    #print('C5:', q - (CHp1[1:]@CHp1[1:]/self.conf['Mb']))
    #print('kE:',kE)
    #print('prel:',np.linalg.norm(prel))
    ### Record Quantities
    qLab = RB[0] @ (RB[1] @ to4mom(-kRot,0)) #gluon 4mom in lab frame
    x1 = xRec - self.quarkCon[n1].pos
    ### [evtype, p1_0, p1_1, p1_2, p1_3, p2..., q..., k..., x1_1, x1_2, x1_3, CosThetag]
    self.recDump['evLog'].append(['RGR', p1[0], p1[1], p1[2], p1[3], p2[0], p2[1], p2[2], p2[3], qLab[0], qLab[1], qLab[2], qLab[3], klab[0], klab[1], klab[2], klab[3], x1[0], x1[1], x1[2], CosThetag])
    return klab

def recThermal(n1,n2,st,conf):
    return mom324(conf['fBinvCDF_'+st](np.random.rand())*sampleSphere(1)[0], conf['M'+st])
def disThermal(conf):
    return mom324(conf['fBinvCDF_b'](np.random.rand())*sampleSphere(1)[0], conf['Mb']), mom324(conf['fBinvCDF_b'](np.random.rand())*sampleSphere(1)[0], conf['Mb']) 
