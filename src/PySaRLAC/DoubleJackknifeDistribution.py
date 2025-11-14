import numpy as np
import math
from .RawDataDistribution import *
from .JackknifeDistribution import *

class DoubleJackknifeDistribution:
    def __init__(self, N_or_rawdist):
        #if isinstance shape type then 
        if isinstance(N_or_rawdist,RawDataDistribution):
            n = N_or_rawdist.size()
            self.samples = [JackknifeDistribution(n-1) for s in range(n)]
            self.resample(N_or_rawdist)
        else:
            n = N_or_rawdist
            self.samples = [JackknifeDistribution(n-1) for s in range(n)]

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

    def sampleMatrix(self):
        return self.samples
        
    def size(self):
        return len(self.samples)
    
    def shape(self):
        return [len(self.samples), self.samples[0].size()]
        
    def mean(self):
        return np.array([s.mean() for s in self.samples])

    def resample(self,raw):
        n = raw.size()
        sumr = raw.mean()*n

        num = 1./float(n-2);
        for i in range(n):
            for j in range(i):
                self.samples[i][j] = (sumr - raw[i] - raw[j])*num
            jj=i
            for j in range(i+1,n):
                self.samples[i][jj] = (sumr - raw[i] - raw[j])*num
                jj+=1
            assert jj == n-1
                
    #Covariance of the means. covariance(a,a) == a.standardError()**2            
    @staticmethod
    def covariance(a,b):
        assert type(a) == type(b) and isinstance(a,DoubleJackknifeDistribution)
        out = JackknifeDistribution(a.size())
        for s in range(a.size()):
            out[s] = JackknifeDistribution.covariance(a[s],b[s])
        return out
