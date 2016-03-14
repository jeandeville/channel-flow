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

    #A = np.array(A);
    #b = np.array(b);    
    
    NA = len(A);
    Nb = len(b);

   
    U = np.zeros(NA);
     
    if (NA != Nb):
        print("ERREUR DE DIMENSION DANS LA MATRICE A ET b");
    
    else:
        A[0,1]=A[0,1]/A[0,0];
        b[0] = b[0]/A[0,0];
        
        for i in range(1,Nb-1):
            A[i,i+1] = A[i,i+1]/(A[i,i]-A[i,i-1]*A[i-1,i]);
            b[i]=(b[i]-A[i,i-1]*b[i-1])/(A[i,i]-A[i,i-1]*A[i-1,i]);

        b[Nb-1]=(b[Nb-1]-A[Nb-1,Nb-2]*b[Nb-2])/(A[Nb-1,Nb-1]-A[Nb-1,Nb-2]*A[Nb-2,Nb-1]);



        U[Nb-1] = b[Nb-1];
        
        for i in range(1,Nb):
            U[Nb-1-i] = b[Nb-1-i] - A[Nb-1-i,Nb-i]*U[Nb-i];
                

    return U;
                    

