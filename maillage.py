# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:58:08 2016

@author: Said et Jean
"""


import math
import numpy as np

def maillage (h,r,l0):
    
    N0=math.floor(math.log(1+0.5*(r-1)*h/l0)/math.log(r))-1;
    N0 = int(N0);
    lmax = l0*r**N0;
    d = h - l0*((1-r**(N0+1))/(1-r));
    N1 = math.floor(d/lmax);
    N1 = int(N1);
    lcste = d/N1;
    
    y = np.zeros(N0+N1+2);
    
    print(N0, N1);
    
    for i in range(N0+2):
        y[i] =l0 * (1-r**i)/(1-r);
        
    for i in range(N0+2,N0+2+N1):
        y[i] = y[N0+1] + (i-N0-1)*lcste;
        
        
    return y;