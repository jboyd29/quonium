# config.py
#
import time

import numpy as np

def parse_string(input_string):
    try:
        # Try to parse as an int 
        result = int(input_string)
    except ValueError:
        try:
            # try parsing as a float
            result = float(input_string)
        except ValueError:
            # keep it as a string
            result = input_string
    return result

class config:
    def __init__(self, filename="../params"):
        self.data = {}
        self.filename = filename
        self.grabParams()
    def __getitem__(self,key):
        return self.data[key]
    def __setitem__(self,key,value):
        self.data[key] = value
    def __len__(self):
        return len(self.data)
    def __str__(self):
        return str(self.data)
    def __iter__(self):
        return iter(self.data)
    def keys(self):
        return self.data.keys()
    def grabParams(self):
        try:
            with open(self.filename, 'r') as file:
                for line in file:
                    if not line.startswith('#'):
                        items = line.strip().split()
                        if len(items) == 2:
                            key, val = items
                            self.data[key] = parse_string(val)
        except FileNotFoundError:
            print(f"File '{self.filename}' not found.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    def echoParams(self):
        for key in self.data.keys():
            print(key,':',self.data[key])


class stepClock:
    def __init__(self, NSteps):
        self.NSteps = NSteps
        self.initTime = time.time()
        self.startTime = time.time()
        self.stopTime = time.time()
        self.stepTimes = []
        self.totTimes = []
    def start(self):
        self.startTime = time.time()
    def stop(self):
        self.stopTime = time.time()
        # Add last step time difference to list
        self.stepTimes.append(self.stopTime - self.startTime)
        # Update total time elapsed
        self.totTimes.append(self.stopTime - self.initTime)
    def getStepTime(self):
        return self.format_Ts(self.stepTimes[-1])
    def getElapTime(self):
        return self.format_Ts(self.totTimes[-1])
    def getExpectTime(self):
        return self.format_Ts(np.mean(self.stepTimes)*(self.NSteps - len(self.stepTimes)))
    def format_Ts(self, seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        fractional_seconds = int((seconds - int(seconds))*100)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{fractional_seconds:02d}"

class conOutput:
    def __init__(self, LabelSpacing = [['t [fm/c]',12,None],['Nhid/Ntot',24,None],['#D',4,(200,50,50)],['#R',4,(100,100,200)],['StepTime',12,None],['ExpTime',12,None]]):
        self.LS = LabelSpacing #Spacing for console ouput
    def printHeader(self):
        print(" ".join([Sp[0][:Sp[1]].ljust(Sp[1]," ") for Sp in self.LS]))
        print("="*(len(self.LS)-1+sum([Sp[1] for Sp in self.LS])))
    def printLine(self, data):
        lineItems = [str(data[i])[:self.LS[i][1]].ljust(self.LS[i][1]," ") for i in range(len(self.LS))]
        for i in range(len(lineItems)):
            if self.LS[i][2] == None:
                continue
            else:
                lineItems[i] = colr(lineItems[i],self.LS[i][2])
        print(" ".join(lineItems))

def colr(text, color):
    return '\033[38;2;'+str(color[0])+';'+str(color[1])+';'+str(color[2])+'m'+text+'\033[0m'
