import math
import numpy as np

def effectSize(N1,N2,t):
    "Calculate D prime. Typical values for small, medium, and large are 0.2 0.5 0.8"
    return t*math.sqrt((N1+N2)/(N1*N2))

def SampleSize(t,d):
    "Estimates required sample size for given d-prime and t statistic"
    return 2.0/( (d/t)**2 )

def estimateTheta(N,k,prior=np.arange(0,1,0.01)):
    "Use Baye's rule to estimate proportion of sucess from measurement"
    likelihood=np.power(prior,k)*np.power(1-prior,N-k);
    return prior[likelihood.argmax()]