from .JackknifeDistribution import *
from .DoubleJackknifeDistribution import *
from .BootJackknifeDistribution import *
from .CorrelationFunction import *

#Covariance of the means. covariance(a,a) == a.standardError()**2
def covariance(a, b):
    assert a.size() == b.size()
    if isinstance(a,JackknifeDistribution) and isinstance(b,JackknifeDistribution):
        N=a.size()
        avg_a = a.mean()
        avg_b = b.mean()
        v = np.dot( (a.sampleVector()-avg_a), (b.sampleVector()-avg_b) )        
        return v*float(N-1)/float(N);
    elif type(a) == type(b) and isinstance(a,(DoubleJackknifeDistribution,BootJackknifeDistribution)):
        out = type(a[0])(a.size())
        for s in range(a.size()):
            out[s] = covariance(a[s],b[s])
        return out

    #For CorrelationFunction return the covariance *matrix*
    elif isinstance(a, CorrelationFunction) and isinstance(b, CorrelationFunction):
        T = a.size()
        out = [[None for _ in range(T)] for _ in range(T)]
        for t1 in range(T):
            for t2 in range(t1+1):
                out[t1][t2] = covariance(a.value(t1),b.value(t2))
                if t1 != t2:
                    out[t2][t1] = out[t1][t2]
        return out
    
    else:
        assert 0

        
def distEval(dist, some_lambda):
    if(isinstance(dist, list)):
        n = dist[0].size()
        out = type(dist[0])(n)
        for s in range(n):
            ds = [ dist[i][s] for i in range(len(dist)) ]            
            out[s] = some_lambda(ds)
        return out
    else:
        out = copy.deepcopy(dist)
        n = dist.size()
        for s in range(n):
            out[s] = some_lambda(dist[s])
        return out
