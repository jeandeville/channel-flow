# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:29:13 2016
@author: Said Ouhamou et Jean Deville,ISAE Supaéro 2016

"""

###############################################################################
#   COUCHE LIMITE TURBULENTE
#   0 =- 1/rho*dP/dx +d/dy(nu.dU/dy+nut.dU/dy)
#   Calcul de la solution obtnue par un modèle longeur de mélange de Praudlt
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from modules import *


def initialisation (y,h,Rstar,gradient):
      
      
    A=26.0; 
    ########Paramètres#############
    #h: taille de la moitie du canal
    #gradient: gradient de pression modifié (1/rho * dPe/dx)
    #Rstar: nombre de rynolds (Rstar=u_tau*h/nu)
    ###############################
    u_tau=np.sqrt(h*(-gradient));# vitesse caractéristique de frottement (u_tau=mu.dU/dy|y=0) 
    nu=u_tau*h/Rstar; # viscosité dynamique à 0 C
    kappa=0.41;
     
    yplus=Rstar/h*y;
    
    l_melange = kappa * np.multiply(y,1-np.exp(-yplus/A));    
    
    dUdY = np.multiply(0.5*np.power(l_melange,-2),-nu+np.sqrt(nu**2+4.0*u_tau**2*np.multiply(np.power(l_melange,2),(1-y/h)))); 
    dUdY[0] = u_tau**2/nu;
    
    nu_t = np.multiply(np.power(l_melange,2),dUdY);
    nu_t[0] = 0;
            
    k=1.0/0.3*np.multiply(nu_t,dUdY);
    k[0] = 0;
    
    omega= np.multiply(k,np.power(nu_t,-1));
    omega[0] = omega[1];
    Nmax=np.size(y)-1;# nombre de cellules 
    dy = np.zeros(Nmax);
    for i in range(Nmax):
        dy[i] = y[i+1] - y[i];
    
    u=np.zeros(np.size(y));
    u[0]=0.0;
    for i in range(Nmax):
        u[i+1]=dy[i]*dUdY[i]+u[i];
    u[Nmax]=u[Nmax-1];
    
    init=np.zeros([4,np.size(y)]);
    init[0,:]=u;
    init[1,:]=nu_t;
    init[2,:]=k;
    init[3,:]=omega;
    
    
    return init;
