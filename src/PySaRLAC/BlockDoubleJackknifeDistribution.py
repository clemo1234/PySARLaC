import numpy as np
import math
from .RawDataDistribution import *
from .JackknifeDistribution import *

class BlockDoubleJackknifeDistribution:
    @staticmethod
    def blockCrop(N, block_size):
        nblocks = N // block_size
        n = nblocks*block_size
        return n, nblocks
    
    def __init__(self, N_or_rawdist, block_size : int):
        self.block_size = block_size
        
        if isinstance(N_or_rawdist,RawDataDistribution):            
            n, nblocks = self.blockCrop(N_or_rawdist.size(), block_size)
            self.samples = [JackknifeDistribution(n-block_size) for s in range(nblocks)]
            self.resample(N_or_rawdist)
        else:
            n, nblocks = self.blockCrop(N_or_rawdist, block_size)
            self.samples = [JackknifeDistribution(n-block_size) for s in range(nblocks)]

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
        
    def mean(self):
        return np.array([s.mean() for s in self.samples])

    def resample(self,raw):
        nsample, nblock = self.blockCrop(raw.size(), self.block_size)
        assert self.size() == nblock and self.size()*self.block_size == nsample
        Nsum = np.sum(raw.sampleVector()[:nsample])

        nrm = 1./(nsample - self.block_size - 1)

        for i in range(self.size()):
            bin_start = i*self.block_size
            bin_lessthan = bin_start + self.block_size;

            bsum = np.sum(raw.sampleVector()[bin_start:bin_lessthan])
            vbase_i = Nsum - bsum

            #D_ij = ( N<d> - [ \sum_{k = b*i}^{b*(i+1)} d_k ] - d_j ) / (N-b-1)
            j = 0;
            for j_true in range(bin_start):
                self[i][j] = (vbase_i - raw[j_true]) * nrm;
                j+=1
            for j_true in range(bin_lessthan,nsample):
                self[i][j] = (vbase_i - raw[j_true]) * nrm;
                j+=1
            assert j == nsample - self.block_size
                
    #Covariance of the means. covariance(a,a) == a.standardError()**2            
    @staticmethod
    def covariance(a,b):
        assert type(a) == type(b) and isinstance(a,BlockDoubleJackknifeDistribution)
        out = JackknifeDistribution(a.size())
        for s in range(a.size()):
            out[s] = JackknifeDistribution.covariance(a[s],b[s])
        return out
