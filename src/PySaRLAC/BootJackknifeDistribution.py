import numpy as np
import math
from .RawDataDistribution import *
from .BootstrapDistribution import *
from .JackknifeDistribution import *

class BootJackknifeDistribution:
    def __init__(self, arg1, arg2=None):                
        if isinstance(arg1,RawDataDistribution) and isinstance(arg2,BootstrapResampleTable):
            self.samples = None
            self.resample(arg1,arg2)
        else:
            assert 0

    def __getitem__(self, tp):
        if isinstance(tp, tuple):            
            i, j = tp
            return self.samples[i][j]
        else:
            return self.samples[tp]
    
    def __setitem__(self,tp,val):
        if isinstance(tp, tuple):                  
            i, j = tp
            self.samples[i][j] = val
        else:
            assert isinstance(val,JackknifeDistribution)
            self.samples[i] = val
       
    def size(self):
        return len(self.samples)
        
    def mean(self):
        return np.array([s.mean() for s in self.samples])

    def resample(self,raw,rtable):
        nboot = rtable.Nboot()
        nsample = rtable.Nsample()
        if self.samples == None or len(self.samples) != nboot:
            self.samples = [ JackknifeDistribution(nsample) for b in range(nboot) ]
        
        for b in range(nboot):
            rens = RawDataDistribution(raw.sampleVector()[ rtable[b][:] ])
            self.samples[b].resample(rens)
