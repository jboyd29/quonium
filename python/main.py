# main.py
#

import numpy as np
from config import config
from particles import quark
from particles import bound
from particles import particleList

import matplotlib.pyplot as plt


conf = config()
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

boxes = [particleList(conf) for i in range(10)]
j=0
for boxi in boxes:
    
    for i in range(2000):
        boxi.step()
        boxi.recLine()
    print('Box',j,'complete')
    j+=1

tr = []
for i in range(1000):
    res = []
    for boxi in boxes:
        res.append(boxi.rec[i])
    tr.append(np.mean(np.array(res),axis=0))

X = [line[0] for line in tr]
Y = [line[1] for line in tr]

plt.plot(X,Y)
plt.xlabel('t')
plt.ylabel('Nb_bound/Nb_total  [AVG]')
plt.title('Scatter Plot of (X, Y) Pairs')
plt.show()  

