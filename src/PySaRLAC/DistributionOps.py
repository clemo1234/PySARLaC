#1. Evaluate a lambda upon every sample of a single distribution
#2. Evaluate a lambda sample wise over a list of distributions. Here the lambda should expect a list of sample values, one for each input distribution
#both return a new distribution
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
