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


    NA = np.size(A[:,1]);
    Nb = np.size(b);


   
    U = np.zeros([Nb,1]);
     
    if (NA != Nb):
        print("ERREUR DE DIMENSION DANS LA MATRICE A ET b");
    
    else:
        for i in range(0,Nb):
            if i == 0:
                A[0,1]=A[0,1]/A[0,0];
                b[0] = b[0]/A[0,0];
                A[0,0] = 1;
            elif i>0 and i<Nb-1:
                A[i,i+1] = A[i,i+1]/(A[i,i]-A[i,i-1]*A[i-1,i]);
                b[i]=(b[i]-A[i,i-1]*b[i-1])/(A[i,i]-A[i,i-1]*A[i-1,i]);
                A[i,i]=1;
            else:
                b[i]=(b[i]-A[i,i-1]*b[i-1])/(A[i,i]-A[i,i-1]*A[i-1,i]);
                A[i,i]=1;
                
    print(A)
        
    for i in range(Nb):
        if i == 0:
            U[Nb-1] = b[Nb-1];
        else:
            U[Nb-1-i] = b[Nb-1-i] - A[Nb-1-i,Nb-i]*U[Nb-i];
                

    return A;
                    
            
    