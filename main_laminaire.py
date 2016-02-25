# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:15:44 2016

@author: Said
"""

#########################################
#   COUCHE LIMITE LAMINAIRE
#   dU/dt = 1/rho*dP/dx +nu.d2U/dy2
#########################################


import numpy as np
import matplotlib.pyplot as plt
from maillage import *;
from laplacien import *;


#   Paramètres
h = 1 # taille de la moitie du canal
r = 1.0001 # taux de croissance du maillage
Nmax = 10; # taille du maillage
Niter = 10000; # nombre d'itérations
dt = 0.01; # pas de calcul
dy0= 0.01;# pas de la première cellule 
init =10.0; # valeur initiale 
gradient=-1.0; # gradient de pression modifié (1/rho * dPe/dx)
nu=1.75*10**-3; # viscosité dynamique à 0 C

#maillage
y = maillage(h,r,dy0);

#autres paramètres
Nb_Pts = np.size(y);
Nmax = Nb_Pts - 1;

##affichage du maillage brut
#for i in range(Nb_Pts):
#    plt.plot([0.0, 10.0], [y[i], y[i]], 'r-', lw=2) 
#plt.show()

#solution exacte
Uexact = 1/(2*nu)*gradient*np.multiply(y,np.add(y,-2*h));
#plt.plot(y,Uexact)
#plt.show();

# Initialisation
U0 = np.zeros(([Nmax+1,1]));
for t in range(Nmax+1):
    U0[t,0]=Uexact[t]*0.5*0.;

#U0 = 1./2.* Uexact;
#plt.plot(y,U0)
#plt.show();



# matrice laplacien modifiée pour prendre en compte la condition aux limites de symetrie
A = laplacien(y);
A[Nmax,:]=A[Nmax-1,:]; 

# matrice identité modifiée
Id=np.eye(Nmax+1,Nmax+1);
Id[0,0]=0;
Id[Nmax,Nmax]=0;
Id[Nmax,Nmax-1]=1;

# second membre  (C1 * vecteur(0,1,...,1) cf fichier word)
B= dt*gradient*np.ones((Nmax+1,1)); 
B[0,0] = 0;


#fig = plt.figure()
U=U0;
for t in range(Niter):
    Uprecedent=U;
    U=np.dot(np.add(Id,dt*nu*A),Uprecedent)-B;
    plt.clf()
    plt.plot(y,U)
    #plt.plot(y,Uexact)
    plt.show();
    plt.pause(0.0001)
    
    # precision = np.linalg.norm(U-Uprecedent);
    
    
    
    
