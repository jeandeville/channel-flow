# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:03:10 2016

@author: jeandeville
"""

############
#
# Thomas Algorithm for Tridiagonal Matrix
#
############

import math
import numpy as np

def thomas (A,b):


    NA = A.size;
    Nb = b.size;
   
    U = np.zeros([Nb,1]);
     
    if (math.sqrt(NA) != Nb):
        print("ERREUR DE DIMENSION DANS LA MATRICE A ET b");
    
    else:
        for i in range(0,Nb):
            if i == 0:
                A[0,1]=A[0,1]/A[0,0];
                b[0] = b[0]/A[0,0];
                A[0,0] = 1;
            elif i<Nb-1:
                A[i,i+1] = A[i,i+1]/(A[i,i]-A[i-1,i]*A[i-1,i]);
                b[i]=(b[i]-A[i-1,i]*b[i-1])/(A[i,i]-A[i-1,i]*A[i-1,i]);
                A[i,i]=0;
            else:
                b[i]=(b[i]-A[i-1,i]*b[i-1])/(A[i,i]-A[i-1,i]*A[i-1,i]);
                A[i,i]=0;
                
        
        for i in range(Nb):
            if i == Nb-1:
                U[i] = b[i];
            else:
                U[i] = A[i,i+1]*U[i+1];
                
    return U;
                    
            
    