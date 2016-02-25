# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:16:31 2016

@author: jeandeville
"""

#########################################
#   COUCHE LIMITE LAMINAIRE
#   dU/dt =- 1/rho*dP/dx +d/dy [(nu + nu_t) du:dy]
#   dk/dt = nu_t (du/dy)^2 - beta*omega*k + d/dy[(nu + sigma_k * nu_t) dk/dy]
#   domega/dt = gamma (du/dy)^2 - beta*omega^2 + d/dy[(nu + sigma_omega * nu_t) domega/dy]
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

#Solution laminaire
Ulaminaire = 1/(2*nu)*gradient*np.multiply(y,np.add(y,-2*h));


######
#Initialisation (de k0, omega0, u0)
######











######### 
#Mise à jour du vecteur U_n+1:
#U_n+1 = -dt* Gr + (Id +dt * L(nu_t)).U_n
########







##########
#Mise à jour vecteur K
# K_n+1 = (Id + dt * M(Beta_etoile,omega))^-1 * (dt*N(U_n,nu_t_n) + (Id + dt * L(nu, nu_t_n,sigma_k)).K_n )
##########






##########
#Mise à jour vecteur W
# W_n+1 = (Id + dt * M(Beta,omega))^-1 * (dt*N(U_n,gamma) + (Id + dt * L(nu, nu_t_n,sigma_omega)).W )
##########











