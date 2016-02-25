# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:15:44 2016

@author: Said
"""
#########################################
#   COUCHE LIMITE LAMINAIRE
#   dU/dt =- 1/rho*dP/dx +nu.d2U/dy2
#########################################

import numpy as np
import matplotlib.pyplot as plt
from maillage import *;
from laplacien import *;


#   Paramètres
h = 1 # taille de la moitie du canal
r = 1.01 # taux de croissance du maillage
Niter = 300; # nombre d'itérations
dt = 10.; # pas de calcul
dy0= 0.01;# pas de la première cellule 
gradient=-1.0; # gradient de pression modifié (1/rho * dPe/dx)
nu=1.75*10**-3; # viscosité dynamique à 0 C

#création du maillage
y = maillage(h,r,dy0);

# Def Nb_Pts
Nb_Pts = np.size(y);
Nmax = Nb_Pts-1;

#####affichage du maillage brut#####
#for i in range(Nb_Pts): 
#    plt.plot([0.0, 10.0], [y[i], y[i]], 'r-', lw=2) 
#plt.show()

#solution exacte
Uexact = 1/(2*nu)*gradient*np.multiply(y,np.add(y,-2*h));

# Initialisation
U0 = np.zeros(([Nmax+1,1]));
for t in range(Nmax+1):
    U0[t,0]=Uexact[t]*0.5*0.; #initilisation à 0 (ou bien à sol exacte divisée par 2) 

# matrice laplacien 
A = laplacien(y);

# matrice identité modifiée (pour prendre en compte les conditions aux limites, cf rapport word)
Id=np.eye(Nmax+1,Nmax+1);
#Id[Nmax,Nmax]=1;
Id[Nmax,Nmax-1]=-1;

#matrice A tild du membre de gauche (rassemblement des deux matrices préc)
Atild = np.add(Id ,- dt*nu*A);

# vecteur grad x des constantes
gr= dt*gradient*np.ones((Nmax+1,1)); 

U=U0;
for t in range(Niter):
    Uprecedent=U;
    B = np.add(Uprecedent , -gr);
    B[0,0] = 0; #condition aux limites
    B[Nmax,0] = 0; #condition aux limites
    U = np.linalg.solve(Atild, B);
    plt.clf()
    plt.plot(y,U)
    plt.plot(y,Uexact)
    plt.show();
    plt.ylim((0,300))
    plt.pause(0.0001)
    
    # precision = np.linalg.norm(U-Uprecedent);