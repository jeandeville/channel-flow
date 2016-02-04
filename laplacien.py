# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 18:55:28 2016

@author: Said et Jean
"""

import math
import numpy as np

def laplacien (y):
    
    Nb_Pts = np.size(y);
    Nmax = Nb_Pts -1;
    
    dy = np.zeros(Nmax);
    
    for i in range(Nmax):
        dy[i] = y[i+1] - y[i];
        
    A = np.zeros([Nb_Pts,Nb_Pts]);
    
    for i in range (1,Nb_Pts-1): #i va de 1 à Nb_Pts - 2
        dy_moyen = 0.5 * (dy[i-1] + dy[i]);
        A[i,i] = -2/dy_moyen**2;
        A[i,i-1] = 1/dy_moyen**2;
        A[i,i+1] = 1/dy_moyen**2;
        
    return A;