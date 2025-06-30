from .DistributionOps import *
from .CorrelationFunction import *
import copy
import numpy as np

class FitCosh:
    def __init__(self,Lt):
        self.Lt = Lt
    def value(self,t,params):
        assert isinstance(params,(tuple,list,np.ndarray)) and isinstance(params[0],float) and len(params)==2, print(type(params),type(params[0]),len(params))
        return params[0] * ( np.exp(-params[1]*t) + np.exp(-params[1]*(self.Lt-t)) )  #allows both single and array arguments for t
    
    def deriv(self,tvals,params):
        assert isinstance(params,(tuple,list,np.ndarray)) and len(params)==2
        T=len(tvals)
        out = np.zeros((T,2))
        out[:,0] = ( np.exp(-params[1]*tvals) + np.exp(-params[1]*(self.Lt-tvals)) )
        out[:,1] = -tvals*params[0] * np.exp(-params[1]*tvals) - (self.Lt-tvals)*params[0]*np.exp(-params[1]*(self.Lt-tvals))
        return out

    def nparam(self):
        return 2
    
        
class FitConstant:
    def __init__(self):
        pass    
    def value(self,t,params):
        assert isinstance(params,(tuple,list,np.ndarray)) and len(params)==1
        return params[0]
        
    def deriv(self,tvals,params):
        assert isinstance(params,(tuple,list,np.ndarray)) and len(params)==1
        T=len(tvals)
        out = np.zeros((T,1))
        for i in range(T):
            out[i] = 1.
        return out
        
        #return np.array([1.0]).T
    def nparam(self):
        return 1

#Evaluate the fitfunc at a coordinate or for each coordinate in a CorrelationFunction
def evaluateFitFunc(ff, arg, params):
    if(isinstance(arg, CorrelationFunction)):
        #Evaluate for all points in input and return correlation function
        out = CorrelationFunction(arg.size())
        for i in range(arg.size()):
            out.setCoord(i,arg.coord(i))
            out.setValue(i, distEval(params, lambda psample : ff.value(arg.coord(i),psample)))
        return out
    else:
        #Assume is coordinate type
        return distEval(params, lambda psample : ff.value(coord,psample))

