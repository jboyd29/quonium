# interpolations.py
#

import numpy as np
import csv
from scipy.interpolate import CubicSpline



# 1D interpolation
class interpol1D:
    def __init__(self,conf,tag):
        self.conf = conf
        self.tag = tag
        self.interp = self.load()
    def __call__(self,x):
        return self.interp(x)
    def load(self):
        data_list = []
        with open('interpolations/'+self.conf[self.tag], 'r', newline='')as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            header = next(reader)
            for row in reader:
                data_list.append(float(row[0]))
        data_array = np.array(data_list)
        return CubicSpline(np.linspace(float(header[0]),float(header[1]),len(data_array)),data_array)

# probability block gets either (a size or a list of blocks) AND a tag to know who it belonged to, 
#potentially you could store the function objects for the results gere instead 
class probBlock:
    def __init__(self, inp, tag):
        self.content=inp
        self.tag = tag
        if type(inp)==float:
            self.size=inp
        elif type(inp)==list:
            self.size=sum([pB.size for pB in inp])
    def __call__(self, r):
        if type(self.content)==float:
            return [self.tag]
        runtot=0
        for pB in self.content:
            runtot+=pB.size
            print(r-runtot)
            if r < runtot:
                return pB(r-runtot)+[self.tag]   

