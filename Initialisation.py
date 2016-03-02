# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:29:13 2016

@author: Said
"""

###############################################################################
#   COUCHE LIMITE TURBULENTE
#   0 =- 1/rho*dP/dx +d/dy(nu.dU/dy+nut.dU/dy)
#   Calcul de la solution obtnue par un modèle longeur de mélange de Praudlt
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from maillage import *
from laplacien import *




def initialisation (y,h,Rstar,gradient):
      
      
      
    ########Paramètres#############
    #h: taille de la moitie du canal
    #gradient: gradient de pression modifié (1/rho * dPe/dx)
    #Rstar: nombre de rynolds (Rstar=u_tau*h/nu)
    u_tau=np.sqrt(h*(-gradient));# vitesse caractéristique de frottement (u_tau=mu.dU/dy|y=0) 
    nu=u_tau*h/Rstar; # viscosité dynamique à 0 C
    kappa=0.41;
     
     

    yplus=Rstar/h*y;
    uplus_1=np.multiply(np.add(-1.0,np.sqrt(np.add(1.0,4.0*np.power(kappa*yplus,2)))),np.power(2.0*kappa**2*yplus,-1))
    uplus_2=(1.0/kappa)*np.log(np.add(np.sqrt(np.add(1.0,4.0*np.power(kappa*yplus,2))),2.0*kappa*yplus));
    uplus=np.add(uplus_1,uplus_2);
    u=u_tau*uplus
    u[0] = 0;
    

    l_melange = kappa * y;    
    
    dUdY = np.multiply(0.5*np.power(l_melange,-2),-nu+np.sqrt(nu**2+4.0*u_tau**2*np.multiply(np.power(l_melange,2),(1-y/h)))); 
    dUdY[0] = u_tau**2/nu;
    
    
    nu_t = np.multiply(np.power(l_melange,2),dUdY);
    nu_t[0] = 0;
    k=1.0/0.3*np.multiply(nu_t,dUdY);
    k[0] = 0;
    omega= np.multiply(k,np.power(nu_t,-1));
    omega[0] = omega[1];
    
    
    
    
    init=np.zeros([4,np.size(y)]);
    init[0,:]=u;
    init[1,:]=nu_t;
    init[2,:]=k;
    init[3,:]=omega;
    
    
    return init;
    
    
    
    
    #plt.plot(yplus,uplus)
    #plt.plot(y,u)
    
    #plt.semilogx(yplus,uplus)
    #plt.xlim((1.0,1000))
    #plt.show();
    
    
    
    
    
    
    
    
    
