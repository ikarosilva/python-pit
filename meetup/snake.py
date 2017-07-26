'''
Created on Jul 24, 2017

@author: isilva
'''


def rank(n):
    p={1:[],2:[],3:[],4:[],5:[]};
    count=1
    b=1
    for x in range(1,n+1):
        p[count].append(x)
        print "id=%s pool=%s seed=%s"%(x,count,len(p[count]))
        if(x != n):
            count=count+b
            if(count==6):
                b=-1
                count=5
            elif(count==0 and b==-1):
                b=1
                count=1
            else:
                pass
    return len(p[count-b])


if __name__ == "__main__":
    
    n=25
    print rank(n)