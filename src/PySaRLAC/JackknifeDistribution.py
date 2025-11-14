import numpy as np
import math
import copy
from .RawDataDistribution import *

class JackknifeDistribution:
    def __init__(self, arg1, arg2=None):
        if isinstance(arg1,RawDataDistribution) and arg2 == None:
            self.samples = np.zeros(arg1.size())
            self.resample(arg1)
        elif isinstance(arg1,int) and arg2 == None:
            self.samples = np.zeros(arg1)
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
        out = JackknifeDistribution(self.size())
        out.samples = self.samples + r.samples
        return out
    
    def __sub__(self, r):
        out = JackknifeDistribution(self.size())
        out.samples = self.samples - r.samples
        return out
    
    def __mul__(self, r):
        
        if isinstance(r, (float, int)):
            out = JackknifeDistribution(self.size())
            out.samples = self.samples * r
            #return out
        else:
            out = JackknifeDistribution(self.size())
            out.samples = self.samples * r.samples
        
        return out
        
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
        

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
        return np.std(self.samples) * math.sqrt(float(len(self.samples)-1))

    def resample(self,raw):
        n = raw.size()
        mu = raw.mean()
        self.samples = (n*mu - raw.sampleVector())/(n-1)

    #Covariance of the means. covariance(a,a) == a.standardError()**2
    @staticmethod
    def covariance(a,b):
        assert type(a) == type(b) and isinstance(a,JackknifeDistribution)
        N=a.size()
        avg_a = a.mean()
        avg_b = b.mean()
        v = np.dot( (a.sampleVector()-avg_a), (b.sampleVector()-avg_b) )        
        return v*float(N-1)/float(N);

        
