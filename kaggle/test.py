'''
Created on Oct 5, 2017

@author: ikaro
'''
import math
import numpy as np
from numpy import arange, reciprocal, Inf,log, divide, loadtxt, inf, zeros,array,\
    NaN, exp
from scipy.stats import entropy
from matplotlib.pyplot import plot, show,  figure, legend, ylabel, xlabel,\
     fill_between, bar, scatter, ylim, text,title
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from numpy import Inf
from numpy import loadtxt, cumsum

def percentile(theta, pdf):
    
    per5=NaN
    per95=NaN 
    bottom=0
    top=0
    for ind in range(len(theta)):
        bottom+= pdf[ind] 
        if(bottom>0.05):
            per5 = theta[ind]
            break
    for ind in range(len(theta)):
        top+= pdf[-(ind+1)]
        if(top>0.05):
            per95 = theta[-(ind+1)]
            break
    return per5, per95


def estimateTheta(N,best):
    "Use Baye's rule to estimate proportion of success from measurement"
    M=100 #Number of points to estimate in the likelihood function
    theta=arange(best,1,(1-best)/float(M))
    pdf=reciprocal(theta)**N
    pdf=pdf/sum(pdf)
    return theta,pdf
    
def estimatePhi(best,N):
    theta,pdf=estimateTheta(N,best)
    mu = sum(pdf*theta) #Estimate of the mean of the likelihood
    return mu-best
  
import sys 

print "Running..."
theta,pdf=estimateTheta(min([81,3658]), best=0.868)
#theta,pdf=estimateTheta(min([75,3658]), best=0.831)
phi=estimatePhi(0.868, 81)
per=percentile(theta, pdf)
print "theta=%s pdf =%s perc=%s phi=%s"%(theta,pdf,str(per),phi)

sys.exit()

# Analysis of performance on PhysioNet Datasets 
#M=number of test records, N= number of team submissions
chal=[{"best":0.83,"M":15, "N":Inf,'title':'2010 Filling The Gap'}, 
      {"best":0.905,"M":2, "N":Inf,'title':'2004 ST Changes'},
      {"best":0.5353,"M":4000, "N":21,'title':'2012 Mortality Prediction'},
      {"best":0.8426,"M":500, "N":41,'title':'2015 False Alarm'},
      {"best":0.932,"M":500, "N":49,'title':'2011 ECG Quality'},
      {"best":0.9991,"M":22, "N":Inf,'title':'0 MIT-BIH Detection'}
      ]
#Estimate likelihoods using the method previously described
figure(figsize = (12,7))
for x in chal:
    n=min(x['N'],x['M'])
    theta,pdf=estimateTheta(n,x['best'])
    plot(theta,pdf,label=x['title'])
legend() 
xlabel('Top Score')
ylabel('Likelihood')
title('Estimated Top Score Likelihood')

chal=[{"best":0.79,"M":Inf, "N":10,'title':'2001 Afib Prediction'},
          {"best":0.82,"M":Inf, "N":10,'title':'2002 RR Modeling'},
          {"best":0.83,"M":Inf, "N":15,'title':'2010 Filling The Gap'},
          {"best":0.905,"M":Inf, "N":2,'title':'2004 ST Changes'},
          {"best":0.9262,"M":35,"N":13,'title':'2000 Sleep Apnea'},
          {"best":0.92,"M":Inf, "N":21,'title':'2008 T Alternans'}, 
          {"best":0.93,"M":Inf, "N":19,'title':'2009 Hypo Prediction'}, 
          {"best":0.5353,"M":4000, "N":20+1,'title':'2012 Mortality Prediction'}, 
          {"best":1-1/float(log(16.34)),"M":Inf, "N":28,'title':'2006 QT Interval'},
          {"best":0.8426,"M":500, "N":38+3,'title':'2015 False Alarm'}, 
          {"best":0.932,"M":500, "N":49,'title':'2011 ECG Quality'}, 
          {"best":0.879,"M":300, "N":60+1,'title':'2014 Beat Detection'}, 
          {"best":0.97,"M":Inf, "N":20,'title':'2004 Afib Termination'},
          {"best":0.9991,"M":Inf, "N":22,'title':'0 MIT-BIH Beat Detection'}
          ]
N=[]
max_hat=[]
for x in chal:
    n=min(x['N'],x['M'])
    max_hat.append(estimatePhi(x['best'],n)) #Estimate Chance of improvement
    N.append(n)

figure(figsize = (12,7))
scatter(N,max_hat)
ylim((0,0.1))
ylabel("Chance of Improvement ( Phi )")
xlabel("Estimated N from number of submissions or test records")
LABELS=[]
for ind,x in enumerate(chal):
    LABELS.append(x['title'])
    text(N[ind]+1,max_hat[ind],' '.join([x for x in chal[ind]['title'].split(' ')[1:]]))

title('Estimated Chance of Improvement vs N')
show() 


#Kaggle data for seizure prediction ( cummulative number of entries, top score)
data=[(0,0.5),
      (2,0.513430270985),
      (13,0.534266358229),
      (15,0.630684071381),
      (22,0.650499008592),
      (42,0.663759087905),
      (61,0.695571711831),
      (125,0.735292465301),
      (173,0.756934897555),
      (276,0.790631196299),
      (397,0.792514871117),
      (400,0.800789821547),
      (428,0.8399322538),
      (504,0.8399322538)]


N=[x[0] for x in data]
score=[x[1] for x in data]
figure(figsize = (12,7))
plot(N,score,'b-o',label='Current Top Score')
M=len(score)
best=zeros((M,1))
pred=zeros((M,1))
pred_range=zeros((M,2))

opt=0
for ind in range(M):
    opt=score[ind] if score[ind]>opt else opt
    best[ind]=opt
    n=min([7,ind])
    theta, pdf=estimateTheta(n,opt)
    mu = sum(pdf*theta)
    pred[ind]=mu
    per5, per95 = percentile(theta, pdf)
    pred_range[ind,:]=[per5,per95]

plot(N,pred,'r',label='Predicted Running Average Top Score')
fill_between(N, pred_range[:,0], pred_range[:,1], color='r', alpha=0.2,
             label='95% Confidence Interval')
legend(loc='lower right')
ylabel('Top Score')
xlabel('Cumulative Number of Entries')
title('Running Estimate of Top Score Limits for Seizure Prediction')
show()