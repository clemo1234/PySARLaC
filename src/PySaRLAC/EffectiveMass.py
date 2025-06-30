from .CorrelationFunction import *
from .FitFunc import *
import numpy as np
import scipy.optimize as opt

class EffMassCosh:
    def __init__(self,Lt):
        self.ff = FitCosh(Lt)
    def value(self,t,m):
        p = (1., m)
        return self.ff.value(t,p) / self.ff.value(t+1,p)
    
    def deriv(self,tvals,m):
        p = (1., m)
        
        vt = self.ff.value(tvals,p)
        dt = self.ff.deriv(tvals,p)
        tp = tvals + 1
        
        vtp = self.ff.value(tp,p)
        dtp = self.ff.deriv(tp,p)

        T = len(tvals)
        out = np.zeros((T,1))
        for t in range(T):
            out[t] = dt[t,1] / vtp - vt / vtp**2 * dtp[t,1]
        return out

    def nparam(self): 
        return 1


def effectiveMass(corfunc, effmass_fitfunc):
    N = corfunc.value(0).size()
    Lt = corfunc.size()
    out = CorrelationFunction(Lt-1)
    dist_type = type(corfunc.value(0))

    ff = lambda xx,m : effmass_fitfunc.value(xx,m)
    dd = lambda xx,m : effmass_fitfunc.deriv(xx,m)
                
    for t in range(Lt-1):
        x = corfunc.coord(t)
        out.setCoord(t,x)
        out.setValue(t, dist_type(N))
        
        for s in range(N):
            y = corfunc.value(t)[s]/corfunc.value(t+1)[s]
            
            r = opt.curve_fit(ff, [x], [y], p0=[0.5], jac=dd)
            out.value(t)[s] = r[0][0]
    return out
