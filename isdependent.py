#! /usr/bin/env python3                                                                                   
# Script for generate error matrix and estimating the number                                              
# of unique entries based on pair-wise indepence of the error                                             
# probability. Records are rows and entries are column in the                                             
# CSV file matrix.txt. The first column is the alarm name,                                                
# (converted to NaN) and the second column is the gold standard.                                          


import numpy as np
import sys
from scipy.stats import beta

print("Generating error matrix...")
data=np.genfromtxt("matrix.txt",delimiter=",")
data=(data.T - data[:,1]).T
data = data !=0
data=data[:,2:]

np.savetxt("ERR.txt",data, delimiter=",",fmt='%i')
print("***Finished generating error matrix...", file=sys.stderr)
print("***Calculating number of independent entries....")

#First entry is independent                                                                               
N=data.shape[0]
M=data.shape[1]
#Confidence interval for detecting dependent entries                                                      
ci=1-(0.000001/M)
corr_matrix=np.zeros(shape=(M,M),dtype=np.int8)
count=1

#Loop through each entry (column) and check if the probability                                            
#of error for that entry is pair-wise independent of the probability of                                   
#error from all previously independent entries                                                            
ind_list=data.copy()
#ind_list=ind_list[:,1:]                                                                                  
for col, current_entry in enumerate(data.T):
    #Get probability of error along with 95% confidence interval                                          
    is_ind=1
    k=sum(current_entry)
    p1=k/N
    myBeta1=beta(1+k,N-k+1)
    intrv1=myBeta1.interval(ci)
    print(k,p1,N,ind_list.shape)
    rm_ind=list()
    for col2, test_entry in enumerate(data.T):
        #Get probability of error, given that the second entry errored                                    
        if col==col2:
            continue
        try:
            N2=sum(test_entry)
        except:
            print(test_entry)
            raise

        k2=sum(current_entry[test_entry !=0])
        p2=k2/N2
        myBeta=beta(1+k2,N2-k2+1)
        intrv2=myBeta.interval(ci)
        print('[%u %u] [%f %f] [%f %f] p1=%f p2=%f N2=%u'
               % (col,col2,intrv1[0],intrv1[1],intrv2[0],intrv2[1],p1,p2,N2))
        if ( (intrv1[1]<intrv2[0]) or (intrv1[0]>intrv2[1]) ):
            #if( p1 <intrv2[0] or p1>intrv2[1]):                                                          
            is_ind=0
        #else:                                                                                            
            #rm_ind.append(col2)                                                                          

    ind_list=np.delete(ind_list,rm_ind)
    if is_ind==1:
        count=count+1

print('Found= %u independent entries from %u (ci = %f) ' % (count, M,ci))
