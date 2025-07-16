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
        assert isinstance(params,(tuple,list,np.ndarray)) and len(params)==1, "Invalid params " + str(type(params)) + " len " + str(len(params))
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



class FitMulti:
    def __init__(self):
        self._nparam = 0
        self.fitfuncs = []
        self.param_indices = []

    def nparam(self):
        return self._nparam

    #Add a fitfunc instance to the multifit. param_indices should be an array indicating the mapping between the parameters of this function to the parameters of the multifit function
    def addFitFunc(self,fitfunc, param_indices):
        assert isinstance(param_indices,list)
        assert len(param_indices) == fitfunc.nparam()
        self.fitfuncs.append(fitfunc)
        self.param_indices.append(param_indices)
 
        for p in param_indices:
            if p >= self._nparam:
                self._nparam = p+1
    
    def _splitCoords(self,coord):
        nfitfunc = len(self.fitfuncs)
        splt = [ [] for i in range(nfitfunc) ]           
        splt_map = [ [] for i in range(nfitfunc) ]
        for c in range(len(coord)):
            fidx, t = coord[c]
            fidx=int(fidx) #scipy converts everything to floats!!!!WHYWHYWHY???@?
            splt[fidx].append(t)            
            splt_map[fidx].append(c)
        for fidx in range(nfitfunc):
            splt[fidx] = np.array(splt[fidx])
        
        return splt, splt_map

    def _getParams(self,params,fidx):
        return params[self.param_indices[fidx]]

    #coordinates are expected to be tuple(fitfunc_idx, fitfunc_coord) or a list/array thereof
    def value(self,coord, params):
        if isinstance(params,(tuple,list)): #ensure we can slice it
            params = np.array(params)
        
        if isinstance(coord,tuple):
            fidx, t = coord
            psub = self._getParams(params,fidx)
            return self.fitfuncs[fidx].value(t, psub)
        elif isinstance(coord, (list,np.ndarray)) and isinstance(coord[0],(tuple,np.ndarray)):
            out = np.zeros(len(coord))
            nfitfunc = len(self.fitfuncs)
            splt, splt_map = self._splitCoords(coord)

            for fidx in range(nfitfunc):
                if len(splt_map[fidx]) > 0:                    
                    psub = self._getParams(params,fidx)
                    t = splt[fidx]
                    vals = self.fitfuncs[fidx].value(t, psub)
                    out[ splt_map[fidx] ] = vals
            return out
        else:
            print("Invalid type "+ str(type(coord)))
            assert 0
            
    def deriv(self,coord,params):
        if isinstance(params,(tuple,list)):
            params = np.array(params)
        
        assert isinstance(coord, (list,np.ndarray)), str(type(coord))
        assert isinstance(coord[0],(tuple,np.ndarray)), str(type(coord[0]))
        out = np.zeros( (len(coord), self._nparam) )
        
        nfitfunc = len(self.fitfuncs)
        splt, splt_map = self._splitCoords(coord)

        for fidx in range(nfitfunc):
            if len(splt_map[fidx]) > 0:
                psub = self._getParams(params,fidx)
                t = splt[fidx]                
                ders = self.fitfuncs[fidx].deriv(t, psub)
                for ii in range(len(splt_map[fidx])):
                    i = splt_map[fidx][ii]
                    for jj in range(len(self.param_indices[fidx])):
                        j = self.param_indices[fidx][jj]
                        out[i,j] = ders[ii,jj]
        return out

    
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

