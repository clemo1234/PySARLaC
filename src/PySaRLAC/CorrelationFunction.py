import numpy as np

class CorrelationFunction:
    def __init__(self,Lt):
        self.resize(Lt)
    def resize(self,Lt):
        self.coords = [None for t in range(Lt)]
        self.values = [None for t in range(Lt)]
    def size(self):
        return len(self.coords)
        
    def setCoord(self,t,v):
        self.coords[t] = v        
    def coord(self,t):
        return self.coords[t]
    def setValue(self,t,v):
        self.values[t] = v
    def value(self,t):
        return self.values[t]

    #Return a CorrelationFunction that has been resampled to the provided type
    def resample(self, dist_type, resample_args=None):
        Lt = self.size()
        out = CorrelationFunction(Lt)
        for t in range(Lt):
            out.setCoord(t, self.coord(t))
            if resample_args != None:
                out.setValue(t, dist_type(self.value(t),resample_args))
            else:
                out.setValue(t, dist_type(self.value(t)))
        return out

    #Return a new CorrelationFunction containing data in the specified index range
    def sliceRange(self, idx_start, idx_end):
        sz_out = idx_end - idx_start + 1
        out = CorrelationFunction(sz_out)
        tt=0
        for t in range(idx_start,idx_end+1):
            out.setCoord(tt, self.coord(t))
            out.setValue(tt, self.value(t))            
            tt+=1
        return out

    #Return a new Correlation function containing data satisfying a condition based on the coordinate, input as a lambda (or single argument function)
    #e.g. cor.subset(lambda c : True if c >= fit_start and c <= fit_end else False)
    def subset(self, where_lambda):
        keep = []
        for i in range(self.size()):
            if where_lambda(self.coord(i)) == True:
                keep.append(i)
        out = CorrelationFunction(len(keep))
        for i in range(len(keep)):
            out.setCoord(i, self.coord(keep[i]))
            out.setValue(i, self.value(keep[i]))  
        return out
    
    #Generic fold routine. 
    #The data is symmetric as   C(Lt - fold_offset - t) ~ C(t) 
    #e.g. fold_offset should be  2*tsep_pipi  for pipi2pt,   tsep_pipi  for pipi->sigma and 0 for sigma 2pt
    def fold(self, fold_offset=0):
        Lt=self.size()
        out = CorrelationFunction(Lt)
        Tref = Lt-fold_offset
        for t in range(Lt):
            out.setCoord(t,self.coord(t))
            out.setValue(t, ( self.value(t) + self.value( (Tref-t+Lt) % Lt ) )/2. )
        return out

    #Product the x-axis, y-axis and y-error for plotting purposes
    def plotInputs(self):
        x = np.array([self.coord(i) for i in range(self.size()) ])
        mu = np.array([ self.value(i).mean() for i in range(self.size()) ])
        sigma = np.array([ self.value(i).standardError() for i in range(self.size()) ])
        return x,mu,sigma

    def __str__(self):
        out = ""
        for i in range(self.size()):
            out += str(self.coord(i)) + " : " + str(self.value(i)) + "\n"
        return out
    
