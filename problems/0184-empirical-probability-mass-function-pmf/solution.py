import numpy as np
def empirical_pmf(samples):

    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    samples=sorted(samples)
    i=0
    res=[]
    while (i<len(samples)):
        c=samples[i]
        j=i
        count=0
        while(j<len(samples))and(samples[j]==samples[i]):
            count+=1
            j+=1
        res.append((samples[i],(count/len(samples))))
        i=j
    return res

    pass