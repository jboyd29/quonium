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
from interpolations import calcRelExpec

import interpolations


import matplotlib.pyplot as plt

import multiprocessing


conf = config()
conf['StateList'] = ['1S']
conf.echoParams()

stColMap = {'b':(0.923, 0.386, 0.209),'1S':(0.368,0.507,0.710),'2S':(0.881,0.611,0.142)}

#box = particleList(conf)
#for i in range(1000):
#    box.step()
#    box.recLine()
#X = [line[0] for line in box.rec]
#Y = [line[1] for line in box.rec]

#plt.plot(X,Y)
#plt.xlabel('t')
#plt.ylabel('Nb_bound/Nb_total')
#plt.title('Bound Fraction')
#plt.show()

Loc_rates = rateMan(conf)
Loc_dists = sampMan(conf)


box = particleList(conf, Loc_rates, Loc_dists)

conOut = conOutput()
conOut.printHeader()

for i in range(conf['tFn']):
    box.step2()
    box.recLine()
    conOut.printLine([box.rec[-1][0], box.rec[-1][1], box.getNDEvs(), box.getNREvs(), box.cl.getStepTime(), box.cl.getExpectTime()])
    
X = [line[0] for line in box.rec]
Y = [line[1] for line in box.rec]




plt.figure(figsize=plt.figaspect(0.5))

plt.subplot(221)
plt.semilogy(X,Y)
plt.axhline(y=calcClassExpec(conf), color='r', linestyle='--', label='Non-relativistic')
plt.axhline(y=calcRelExpec(conf), color='g', linestyle='--', label='Relativistic')
#plt.ylim(0.0001,1)
plt.xlabel('t [GeV-1]')
plt.ylabel('Nb_bound/Nb_total')
plt.title('Hidden bottom fraction')

plt.subplot(222)
plt.hist(box.getMoms(), bins=np.linspace(0,conf['prCut'],conf['NPts']), color = stColMap['b'])
plt.plot(np.linspace(0,conf['prCut'],conf['NPts']),interpolations.getNMomDistPlot(conf,'b')*box.getNunbound()*conf['prCut']/conf['NPts'], color=tuple(x*0.7 for x in stColMap['b']), label='Therm '+'b-quark')
for st in conf['StateList']:
    plt.hist(box.getMomsB(st), bins=np.linspace(0,conf['prCut'],conf['NPts']), color=stColMap[st])
    plt.plot(np.linspace(0,conf['prCut'],conf['NPts']),interpolations.getNMomDistPlot(conf, st)*box.getNbound(st)*conf['prCut']/conf['NPts'], color=tuple(x*0.7 for x in stColMap[st]), label='Therm '+st) 

plt.xlabel('p [GeV]')
plt.ylabel('Count')
plt.legend()
plt.title('Particle momentum distribution')

#plt.subplot(224)
#plt.plot(interpolations.getNMomDistPlot(conf,'b')*box.getNunbound(), color='b')
#plt.xlabel('p')
#plt.ylabel('Count')
#plt.title('')

plt.tight_layout()
plt.show()


exit()

