import numpy as np

class hydroBac:
    def __init__(self, conf):
        self.conf = conf
        self.data = self.readIn()
    def __call__(self, x, t): # x-numpy 3vec , t-float
        #return self.data[t][x.T[0]][x.T[1]][x.T[2]]
        ### FOR now until i fiugre out the hydro representation this will always return 0
        return np.zeros((len(x),3))
    def readIn(self):
        if conf['HydroMode'] == 0: #Static case
            shape = (self.conf['tFn'], self.conf['HPts'], self.conf['HPts'], self.conf['HPts'])
            return np.zeros(shape)

