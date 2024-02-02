# main.py
#

import numpy as np

from config import config
from config import conOutput
from particles import quark
from particles import bound
from particles import particleList

from interpolations import rateMan
from interpolations import sampMan

from interpolations import calcClassExpec
from interpolations import calcRelExpec2

from interpolations import getRGAratePlot

import interpolations


import matplotlib.pyplot as plt

import multiprocessing


conf = config()
conf['StateList'] = ['1S']
conf.echoParams()

stColMap = {'b':(0.923, 0.386, 0.209),'1S':(0.368,0.507,0.710),'2S':(0.881,0.611,0.142)}


Loc_rates = rateMan(conf)
Loc_dists = sampMan(conf)

def runBox(conf, Lrates, Ldists):
    box = particleList(conf, Lrates, Ldists)
    conOut = conOutput()
    conOut.printHeader()
    for i in range(conf['tFn']):
        box.step2()
        box.recLine()
        conOut.printLine([box.rec[-1][0], box.rec[-1][1], box.getNDEvs(), box.getNREvs(), box.cl.getStepTime(), box.cl.getExpectTime()])
    return np.array([line[1] for line in box.rec]) # <- Hidden fraction values

# Run set of boxes on given initial momentum distribution

conf['pSampleType'] = 2
Sig00_10 = np.array([runBox(conf, Loc_rates, Loc_dists) for i in range(100)])
#Sig00_100 = np.array([runBox(conf, Loc_rates, Loc_dists) for i in range(3)])

#conf['pSampleType'] = 3
#conf['pSampSig'] = 0.2
#Sig02_10 = np.array([runBox(conf, Loc_rates, Loc_dists) for i in range(100)])
#Sig02_100 = np.array([runBox(conf, Loc_rates, Loc_dists) for i in range(100)])

#conf['pSampleType'] = 3
#conf['pSampSig'] = 1.0
#Sig1_10 = np.array([runBox(conf, Loc_rates, Loc_dists) for i in range(100)])
#Sig1_100 = np.array([runBox(conf, Loc_rates, Loc_dists) for i in range(100)])

conf['pSampleType'] = 0
Thermrun = np.array([runBox(conf, Loc_rates, Loc_dists) for i in range(100)]) 

def getLineWBands(dat):
    return np.array([[np.mean(lin), np.std(lin)] for lin in dat.T])

tVs = np.array([(i+1)*conf['dt'] for i in range(conf['tFn'])])
MF = 2

plt.figure(figsize=plt.figaspect(0.5))

plt.subplot(121)

datSig00_10 = getLineWBands(Sig00_10)
plt.semilogy(tVs, datSig00_10[:,0], color = (0.9,0,0), label = 'Pz = 10 GeV : NBoxes = 100')
plt.fill_between(tVs, datSig00_10[:,0] - datSig00_10[:,1]*MF, datSig00_10[:,0] + datSig00_10[:,1]*MF, color = (0.9,0,0), alpha = 0.25)
#datSig00_100 = getLineWBands(Sig00_100)
#plt.plot(tVs, datSig00_100[:,0], color = (0.6,0,0), label = 'Sig = 0 : NBoxes = 100')
#plt.fill_between(tVs, datSig00_100[:,0] - datSig00_100[:,1]*MF, datSig00_100[:,0] + datSig00_100[:,1]*MF, color = (0.6,0,0), alpha = 0.5)

#datSig02_10 = getLineWBands(Sig02_10)
#plt.semilogy(tVs, datSig02_10[:,0], color = (0,0.9,0), label = 'Sig = 0.2 : NBoxes = 100')
#plt.fill_between(tVs, datSig02_10[:,0] - datSig02_10[:,1]*MF, datSig02_10[:,0] + datSig02_10[:,1]*MF, color = (0,0.9,0), alpha = 0.3)
#datSig02_100 = getLineWBands(Sig02_100)
#plt.plot(tVs, datSig02_100[:,0], color = (0,0.6,0), label = 'Sig = 0.2 : NBoxes = 100')
#plt.fill_between(tVs, datSig02_100[:,0] - datSig02_100[:,1]*MF, datSig02_100[:,0] + datSig02_100[:,1]*MF, color = (0,0.6,0), alpha = 0.5)

#datSig1_10 = getLineWBands(Sig1_10)
#plt.semilogy(tVs, datSig1_10[:,0], color = (0,0,0.9), label = 'Sig = 1 : NBoxes = 100')
#plt.fill_between(tVs, datSig1_10[:,0] - datSig1_10[:,1]*MF, datSig1_10[:,0] + datSig1_10[:,1]*MF, color = (0,0,0.9), alpha = 0.25)
#datSig1_100 = getLineWBands(Sig1_100)
#plt.plot(tVs, datSig1_100[:,0], color = (0,0,0.6), label = 'Sig = 1 : NBoxes = 100')
#plt.fill_between(tVs, datSig1_100[:,0] - datSig1_100[:,1]*MF, datSig1_100[:,0] + datSig1_100[:,1]*MF, color = (0,0,0.6), alpha = 0.5)


datTherm = getLineWBands(Thermrun)
plt.semilogy(tVs, datTherm[:,0], color = (0,0,0.9), label = 'Thermal : NBoxes = 100')
plt.fill_between(tVs, datTherm[:,0] - datTherm[:,1]*MF, datTherm[:,0] + datTherm[:,1]*MF, color = (0,0,0.9), alpha = 0.25)
#datSig1_100 = getLineWBands(Sig1_100)
#plt.plot(tVs, datSig1_100[:,0], color = (0,0,0.6), label = 'Sig = 1 : NBoxes = 100')
#plt.fill_between(tVs, datSig1_100[:,0] - datSig1_100[:,1]*MF, datSig1_100[:,0] + datSig1_100[:,1]*MF, color = (0,0,0.6), alpha = 0.5)

plt.legend()
plt.xlabel('t [GeV-1]')
plt.ylabel('Nhid/Ntot')
plt.title('Hidden fraction initial momentum comparison (log)')


plt.subplot(122)

plt.plot(tVs, datSig00_10[:,0], color = (0.9,0,0), label = 'Pz = 10 GeV : NBoxes = 100')
plt.fill_between(tVs, datSig00_10[:,0] - datSig00_10[:,1]*MF, datSig00_10[:,0] + datSig00_10[:,1]*MF, color = (0.9,0,0), alpha = 0.25)

plt.plot(tVs, datTherm[:,0], color = (0,0.7,0.7), label = 'Thermal : NBoxes = 100')
plt.fill_between(tVs, datTherm[:,0] - datTherm[:,1]*MF, datTherm[:,0] + datTherm[:,1]*MF, color = (0,0,0.9), alpha = 0.25)

plt.xlabel('t [GeV-1]')
plt.ylabel('Nhid/Ntot')
plt.title('Hidden fraction initial momentum comparison (linear)')
plt.legend()
plt.show()


exit()

