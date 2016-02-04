# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:15:44 2016

@author: Said et Jean
"""

#########################################
#   COUCHE LIMITE LAMINAIRE
#   0 = -1/rho*dP/dx +nu.d2U/dy2
#########################################


import numpy as np
import matplotlib.pyplot as plt
from maillage import *;
from laplacien import *;


#   Paramètres
h = 1 # taille de la moitie du canal
r = 1.00001;# taux de croissance du maillage
Nmax = 10; # taille du maillage
Niter = 0; # nombre d'itérations
dt = 1.; # pas de calcul
dy0= 0.001;# pas de la première cellule 
init =10.0; # valeur initiale 
gradient=-1.0; # gradient de pression modifié (1/rho * dPe/dx)
nu=1.75*10**-3; # viscosité dynamique à 0 C

#maillage
y = maillage(h,r,dy0);
#y = np.linspace(0,h,10);
#####affichage du maillage brut#####
#for i in range(Nb_Pts):
#    plt.plot([0.0, 10.0], [y[i], y[i]], 'r-', lw=2) 
#plt.show()

#autres paramètres
Nb_Pts = np.size(y);
Nmax = Nb_Pts - 1;

#solution exacte
Uexact = 1/(2*nu)*gradient*np.multiply(y,np.add(y,-2*h));

# matrice laplacien modifiée pour prendre en compte la condition aux limites de symetrie
A = laplacien(y);
A[Nmax,Nmax]=1;
A[Nmax,Nmax-1]=-1; 
A[0,0] = 1;

# second membre  (C1 * vecteur(0,1,...,1) cf fichier word)
B= gradient*np.ones((Nmax+1,1)); 
B[0,0] = 0;
B[Nmax,0] = 0;

U = np.linalg.solve(nu*A, B);

plt.plot(y,U)
plt.plot(y,Uexact)
plt.show();