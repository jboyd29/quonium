# interpolations.py
#

import numpy as np
import csv
from scipy.interpolate import CubicSpline



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
        
        
                
                
            
            
            
        
        


