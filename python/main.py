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

hbarc = 0.1973 # hbarc = 0.1973 GeV*fm

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
    tot4p = box.getTot4p()
    conOut.printLine([box.rec[-1][0], box.rec[-1][1], box.getNDEvs(), box.getNREvs(), box.cl.getStepTime(), box.cl.getExpectTime(), tot4p[0], tot4p[1], tot4p[2], tot4p[3]])
    
X = np.array([line[0] for line in box.rec])
Y = [line[1] for line in box.rec]


# Write hidden fraction result to file
np.savetxt('../export/HidFrac.tsv', np.array(box.rec), delimiter='\t', fmt='%.8f')



plt.semilogy(X*hbarc,Y)
plt.axhline(y=calcClassExpec(conf), color='r', linestyle='--', label='Non-relativistic')
plt.axhline(y=calcRelExpec2(conf), color='g', linestyle='--', label='Relativistic')
#plt.ylim(0.0001,1)
plt.xlabel('t [fm/c]')
plt.ylabel('Nb_bound/Nb_total')
plt.title('Hidden bottom fraction')
plt.plot()

plt.figure(figsize=plt.figaspect(0.5))


plt.subplot(311)
plt.hist(box.getMoms(), bins=np.linspace(0,conf['prCut'],conf['NPts']), color = stColMap['b'], alpha=0.7)
plt.plot(np.linspace(0,conf['prCut'],conf['NPts']),interpolations.getNMomDistPlot(conf,'b')*box.getNunbound()*conf['prCut']/conf['NPts'], color=tuple(x*0.7 for x in stColMap['b']), label='Therm '+'b-quark')
for st in conf['StateList']:
    plt.hist(box.getMomsB(st), bins=np.linspace(0,conf['prCut'],conf['NPts']), color=stColMap[st], alpha=0.7)
    plt.plot(np.linspace(0,conf['prCut'],conf['NPts']),interpolations.getNMomDistPlot(conf, st)*box.getNbound(st)*conf['prCut']/conf['NPts'], color=tuple(x*0.7 for x in stColMap[st]), label='Therm '+st) 

plt.xlabel('p [GeV]')
plt.ylabel('Count')
plt.legend()
plt.title('Particle momentum distribution')

sampPs = np.linspace(0,conf["prCut"],conf["NPts"]) 
plt.subplot(312)
for st in conf['StateList']:
    plt.plot(sampPs, getRGAratePlot(conf,st,sampPs), color=stColMap[st])
plt.xlabel('p')
plt.ylabel('Rate')
plt.title('Momentum dependent marginal dissociation rates')

RGRp1SDat = interpolations.binRGRdat(conf,box.recDump['RGRrateP1S'])
plt.subplot(313)
plt.plot(RGRp1SDat[:,0],RGRp1SDat[:,1], label='RGR-1S', color = stColMap['b'])
plt.fill_between(RGRp1SDat[:,0], (RGRp1SDat[:,1]-(RGRp1SDat[:,2]*3)), (RGRp1SDat[:,1]+(RGRp1SDat[:,2]*3)), color=stColMap['b'], alpha=0.7)
plt.xlabel('p')
plt.ylabel('Rate')
plt.title('Momentum dependent marginal regeneration rates')


plt.tight_layout()
plt.show()


exit()

