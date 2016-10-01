import math
import numpy as np
from numpy import arange, reciprocal, Inf,log
from scipy.stats import entropy

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

def estimateMax(best,N,max=1,M=100):
        theta=arange(best,max,(max-best)/float(M))
        pdf=reciprocal(theta**N)
        pdf=pdf/sum(pdf)
        mx = sum(pdf*theta)
        return theta,pdf,mx
    
     
     
    
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    chal=[[0.9991, 22],#MIT-BIH 
          [0.9262,13], #2001
          [0.79, 10],#2002
          [0.82, 10],#2002
          [1-1/float(log(3.4)), 8],#2003
          [0.905, 2],#2003
          [0.97, 20],#2004
          [1-1/float(log(16.34)), 28],#2006
          [0.92, 21], #2008
          [0.93, 19 ], #2009
          [0.83, 15 ], #2010
          [0.932, 49 ], #2011
          [0.5353, 20 ], #2012
          [1-1/float(log(187)), 53],#2013
          [0.879, 60], #2014
          [0.8139, 38], #2015 
          ]
    score=[]
    N=[]
    max_hat=[]
    for x in chal:
        theta, pdf,mx=estimateMax(x[0],x[1], max=1, M=100)
        max_hat.append(mx-x[0])
        max_pdf=arange(x[0],1,(1-x[0])/float(100))
        max_ent=entropy(max_pdf/sum(max_pdf))
        score.append(entropy(pdf)/max_ent)
        N.append(x[1])
    plt.scatter(N,score,hold=True)
    plt.figure()
    plt.scatter(N,max_hat)
    plt.ylim((0,max(max_hat)))
    plt.show()
    