import numpy as np
import math
import copy
import random

class RawDataDistribution:
    def __init__(self, arg):
        if(isinstance(arg,int)):
            self.samples = np.zeros(arg)
        elif(isinstance(arg,(list,np.ndarray))):
            N = len(arg)
            self.samples = np.zeros(N)
            for s in range(N):
                self.samples[s] = arg[s]            

    def __getitem__(self,i):
        return self.samples[i]
    def __setitem__(self,i,val):
        self.samples[i] = val

    def __str__(self):
        return "%f +- %f" % (self.mean(), self.standardError())

    def __add__(self, r):
        out = RawDataDistribution(self.size())
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
        return np.std(self.samples)/math.sqrt(float(len(self.samples) - 1))

    def bin(self,bin_size):
        out = []
        nbin = len(vals)//bin_size
        for b in range(nbin):
            off = b*bin_size
            v = 0.
            for i in range(off,off+bin_size):
                v += vals[i]
            v/=bin_size
            out.append(v)

        outd = RawDataDistribution(len(out))
        outd.samples = np.array(out)
            
        return outd

    def randomGaussian(self,mu,sigma):
        for i in range(self.size()):
            self[i] = random.gauss(mu,sigma)
            
