from .CorrelationFunction import *
from .DistributionOps import *
import scipy.optimize as opt
import scipy.linalg as linalg
import numpy as np
import math
import copy

class Fitter:
    def __init__(self, fitfunc):
        self.cov = None
        self.fitfunc = fitfunc

    def generateCovarianceMatrix(self, cor: CorrelationFunction):
        self.cov = CorrelationFunction.covariance(cor,cor)

    #kwargs are passed to curve_fit
    def fit(self, params, cor, **kwargs):
        assert self.cov != None
        n = cor.value(0).size()
        T = cor.size()
        assert len(self.cov) == T

        nparams = self.fitfunc.nparam()
        assert len(params) == nparams        
        for p in params:
            assert p.size() == n
        dof = cor.size() - nparams
        assert dof >= 0
            
        print("Performing a fit with",nparams,"free parameters and",dof,"degrees of freedom")

        chisq_dist = copy.deepcopy(cor.value(0))
        
        for s in range(n):
            scov = np.zeros((T,T))
            for t1 in range(T):
                for t2 in range(t1+1):
                    scov[t1,t2] = self.cov[t1][t2][s]
                    if t2 != t1:
                        scov[t2,t1] = scov[t1,t2]
            sx = [ cor.coord(t) for t in range(T) ]
            sy = [ cor.value(t)[s] for t in range(T) ]

            pguess = [ params[p][s] for p in range(nparams) ]
            
            ff = lambda xx,*pp : self.fitfunc.value(xx,pp)
            dd = lambda xx,*pp : self.fitfunc.deriv(xx,pp)
            
            r = opt.curve_fit(ff, sx, sy, p0=pguess, sigma=scov, jac=dd, absolute_sigma=True,  full_output=True, method='lm', **kwargs) #
            for p in range(nparams):
                params[p][s] = r[0][p]
            
            inv_cov = linalg.inv(scov)
            yfit = np.array([ self.fitfunc.value(sx[i],r[0]) for i in range(T) ])
            ydiff = yfit - np.array(sy)
            chisq = ydiff.T @ inv_cov @ ydiff
            chisq_dist[s] = chisq
        return chisq_dist, dof
