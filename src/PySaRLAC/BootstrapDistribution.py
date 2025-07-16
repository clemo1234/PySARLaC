import numpy as np
import math
import copy
from .RawDataDistribution import *

import numpy as np

class BootstrapResampleTable:
    def __init__(self,nboot, nsample):
        self.rtable = np.zeros( (nboot, nsample) , dtype=np.uint64)
        for b in range(nboot):          
            for s in range(nsample):        
                self.rtable[b][s] = np.random.randint(0,nsample)
    def __getitem__(self,bs):
        return self.rtable[bs]

    def Nboot(self):
        return self.rtable.shape[0]
    def Nsample(self):
        return self.rtable.shape[1]


class BootstrapDistribution:
    def __init__(self, arg1, arg2=None):                
        if isinstance(arg1,RawDataDistribution) and isinstance(arg2,BootstrapResampleTable):
            self.samples = None
            self.resample(arg1,arg2)

        #arg1=Nboot
        elif isinstance(arg1,int) and arg2 == None:
            self.samples = np.zeros(arg1)
            
        #arg1=Nboot, arg2=initial value
        elif isinstance(arg1,int) and isinstance(arg2,float):
            self.samples = np.full(arg1,arg2)
        else:
            assert 0
            
    def __getitem__(self,i):
        return self.samples[i]
    def __setitem__(self,i,val):
        self.samples[i] = val

    def __str__(self):
        return "%f +- %f" % (self.mean(), self.standardError())

    def __add__(self, r):
        out = BootstrapDistribution(self.size())
        out.samples = self.samples + r.samples
        return out

    def __truediv__(self, r):
        if(isinstance(r,float)):
            out = copy.deepcopy(self)
            out.samples = out.samples / r
            return out
        else:
            assert 0    
    
    def sampleVector(self):
        return self.samples
        
    def size(self):
        return len(self.samples)
        
    def mean(self):
        return np.mean(self.samples)

    def standardError(self):
        return np.std(self.samples)

    def resample(self,raw,rtable):
        nboot = rtable.Nboot()
        nsample = rtable.Nsample()
        if self.samples == None or len(self.samples) != nboot:
            self.samples = np.zeros(nboot)
        
        for b in range(nboot):
            rens = raw.sampleVector()[ rtable[b][:] ]
            self.samples[b] = np.mean(rens)


