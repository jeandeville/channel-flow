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
from Initialisation import *

#   Paramètres

#   Paramètres
r = 1.01 # taux de croissance du maillage
dy0= 0.01;# pas de la première cellule 
h = 1 # taille de la moitie du canal
gradient=-1.0; # gradient de pression modifié (1/rho * dPe/dx)
Rstar=100.0; # nombre de rynolds (Rstar=u_tau*h/nu)
u_tau=np.sqrt(h*(-gradient));# vitesse caractéristique de frottement (u_tau=mu.dU/dy|y=0) 
nu=u_tau*h/Rstar; # viscosité dynamique à 0 C
Niter = 300; # nombre d'itérations
dt = 10.; # pas de calcul
####################
####### IL FAUT RAJOUTER LES CONSTANTES SIGMA_K SIGMA_OMEGA ET GAMMA
#######################

#création du maillage
y = maillage(h,r,dy0);

# Def Nb_Pts
Nb_Pts = np.size(y);
Nmax = Nb_Pts-1;

#Def de Gr
Gr = gradient * np.ones(Nb_Pts)

#Solution laminaire
Ulaminaire = 1/(2*nu)*gradient*np.multiply(y,np.add(y,-2*h));


######
#Initialisation (de k0, omega0, u0)
######

Init=initialisation(y,h,Rstar,gradient);
U0=Init[0,:];
nu_t0=Init[1,:];
k0=Init[2,:];
omega0=Init[3,:];

#plt.plot(y,u0)
#print(u0)
#print(k0)
#print(nu_t0)
#print(omega0)

Uprec = U0;
kprec = k0;
omegaprec = omega0;
nu_tprec = nu_t0;
Id = np.eye(Nb_Pts);

################## DEBUT BOUCLE ##############


######### 
#Mise à jour du vecteur U_n+1:
#U_n+1 = -dt* Gr + (Id +dt * L(nu_t)).U_n
########
U = -dt * Gr + np.multiply((Id + dt * L_turbulent(nu,nu_t,1,y)),Uprec); #on envoie 1 car formule de L_Turbulent: nu + sigma*nu_t



##########
#Mise à jour vecteur K
# K_n+1 = (Id + dt * M(Beta_etoile,omega))^-1 * (dt*N(U_n,nu_t_n) + (Id + dt * L(nu, nu_t_n,sigma_k)).K_n )
##########
inverse_matrix = inverse_tdgauche_turbulent (dt, beta_star, omegaprec);
second_terme = dt *  N_turbulent(Uprec,nu_tprec,y) + np.multiply(Id + dt * L_turbulent(nu,nu_tprec,sigma_k,y),kprec);
K = np.multiply(inverse_matrix,second_terme);



##########
#Mise à jour vecteur W
# W_n+1 = (Id + dt * M(Beta,omega))^-1 * (dt*N(U_n,gamma) + (Id + dt * L(nu, nu_t_n,sigma_omega)).W )
##########
inverse_matrix = inverse_tdgauche_turbulent (dt, beta, omegaprec);
second_terme = dt *  N_turbulent(Uprec,gamma,y) + np.multiply(Id + dt * L_turbulent(nu,nu_tprec,sigma_omega,y),omegaprec);
omega = np.multiply(inverse_matrix,second_terme);


##########
#Mise à jour nu_t
# nu_t = k / omega
#########
nu_t = np.multiply(K,np.power(omega,-1));


################## FIN BOUCLE ##############





