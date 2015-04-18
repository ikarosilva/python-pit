import math

def effectSize(N1,N2,t):
    "Calculate D prime. Typical values for small, medium, and large are 0.2 0.5 0.8"
    return t*math.sqrt((N1+N2)/(N1*N2))
