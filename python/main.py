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

import matplotlib.pyplot as plt

import multiprocessing


conf = config()
conf['StateList'] = ['1S']
conf.echoParams()


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
    conOut.printLine([box.rec[-1][0], box.rec[-1][1], box.cl.getStepTime(), box.cl.getExpectTime()])



exit()





def runBox(box,i):
    c=1
    for i in range(conf['tFn']):
        box.step()
        if c%100==0:
            print('B:',i,'  t:',round(box.time,3),'  HidF:',round(box.getOccupationRatio(),3))
        box.recLine()
    result_queue.put([box.rec,box.getMoms()])


#boxes = [particleList(conf, Loc_rates, Loc_dists) for i in range(1)]
#j=0
#moms = []
#for boxi in boxes:
#    
#    for i in range(conf['tFn']):
#        boxi.step()
#        boxi.recLine()
#        #if boxi.time+0.001 % 1 <0.01:
#        print('B:',j,' t:',boxi.time,)
#    moms.append(boxi.getMoms())
#    print('Box',j,'complete')
#    j+=1


NProc=10
result_queue = multiprocessing.Queue()
boxes = [particleList(conf, Loc_rates, Loc_dists) for i in range(NProc)]
processes = []
for i in range(NProc):
    boxi = boxes[i]
    process = multiprocessing.Process(target=runBox, args=(boxi,i))
    processes.append(process)
    process.start()

for process in processes:
    process.join()

results = []
while not result_queue.empty():
    results.append(result_queue.get())
recL=[item[0] for item in results]
moms=[np.linalg.norm(boxL[1][i]) for boxL in results for i in range(len(boxL[1]))]


tr = []
for i in range(conf['tFn']):
    res = []
    for boxL in recL:
        res.append(boxL[i])
    tr.append(np.mean(np.array(res),axis=0))



X = [line[0] for line in tr]
Y = [line[1] for line in tr]


plt.figure(figsize=plt.figaspect(0.25))

plt.subplot(121)
plt.plot(X,Y)
plt.xlabel('t')
plt.ylabel('Nb_bound/Nb_total  [AVG]')
plt.title('Hidden bottom fraction')

plt.subplot(122)
moms = [box.getMoms() for box in boxes]
moms = [item for sublist in moms for item in sublist]
plt.hist(moms, bins=np.linspace(0,conf['prCut']/10,100))
plt.xlabel('p')
plt.ylabel('Count')
plt.title('Quark momentum distribution')


plt.tight_layout()
plt.show()  


