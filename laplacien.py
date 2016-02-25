# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 18:55:28 2016

@author: Said et Jean
"""

import math
import numpy as np
#
## l'ancien laplacien avec la mauvaise dérivée seconde
#def laplacien (y):
#    
#    Nb_Pts = np.size(y);
#    Nmax = Nb_Pts -1;
#    
#    dy = np.zeros(Nmax);
#    
#    for i in range(Nmax):
#        dy[i] = y[i+1] - y[i];
#        
#    A = np.zeros([Nb_Pts,Nb_Pts]);
#    
#    for i in range (1,Nb_Pts-1): #i va de 1 à Nb_Pts - 2
#        dy_moyen = 0.5 * (dy[i-1] + dy[i]);
#        A[i,i] = -2/dy_moyen**2;
#        A[i,i-1] = 1/dy_moyen**2;
#        A[i,i+1] = 1/dy_moyen**2;
#        
#    return A;


def laplacien (y):
    
    Nb_Pts = np.size(y);
    Nmax = Nb_Pts -1;
    
    dy = np.zeros(Nmax);
    
    for i in range(Nmax):
        dy[i] = y[i+1] - y[i];
        
    A = np.zeros([Nb_Pts,Nb_Pts]);
    
    for i in range (1,Nb_Pts-1): #i va de 1 à Nb_Pts - 2
        dy_moyen = 0.5 * (dy[i-1] + dy[i]);
        
        A[i,i] = -2/(dy[i-1]*dy[i]);
        A[i,i-1] = 1/(dy_moyen*dy[i-1]);
        A[i,i+1] = 1/(dy_moyen*dy[i]);
        
    return A;
    
    

def laplacien_turbulent (y, nu_t,nu,sigma): #Le laplacien turbulent n'est pas vraiment un laplacien, il s'agit du terme en dy((nu + sigma*nu_t) du/dy)
    
    Nb_Pts = np.size(y);
    Nmax = Nb_Pts -1;
    
    dy = np.zeros(Nmax);
    
    for i in range(Nmax):
        dy[i] = y[i+1] - y[i];
        
    A = np.zeros([Nb_Pts,Nb_Pts]);
    
    for i in range (1,Nb_Pts-1): #i va de 1 à Nb_Pts - 2
        dy_moyen = 0.5 * (dy[i-1] + dy[i]);
        nu_tsup = 0,5 * (nu_t[i+1] + nu_t[i]) #correspond à "Ci[i+0,5]"
        nu_tinf = 0,5 * (nu_t[i] + nu_t[i-1]) #correspond à "Ci[i-0,5]"
        
        A[i,i] = -(nu+sigma*nu_tsup)/(dy[i+1]*dy_moyen)-(nu+sigma*nu_tinf)/(dy[i-1]*dy_moyen);
        A[i,i-1] = (nu+sigma*nu_tinf)/(dy_moyen*dy[i-1]);
        A[i,i+1] = (nu+sigma*nu_tsup)/(dy_moyen*dy[i]);
        
    return A;